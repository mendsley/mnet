/*
 * Copyright 2015 Matthew Endsley
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted providing that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <mnet/mnet.h>
#include <mnet/platform.h>

#include <deque>
#include <stdio.h>
#include <string.h>
#include "core/allocator.h"
#include "core/memory.h"
#include "core/protocol.h"

namespace mnet {
	namespace {

		static const int64_t c_millisecond           = 1000000;
		static const int64_t c_second                = 1000 * c_millisecond;
		static const int64_t c_connectTimeout        = 300 * c_millisecond;
		static const int64_t c_rentransmitTimeout    = 200 * c_millisecond;
		static const int64_t c_shutdownTimeout       = 3 * c_second;
		static const int64_t c_shutdownPacketTimeout = 400 * c_millisecond;

		static const uint32_t c_maxSequentialRetransmits = 15;
		
		struct ConnectionState {
			enum E {
				closed,
				connecting,
				connect_sendecho,
				connected,
				shutdown_pending,
				shutdown_received,
				shutdown_sent,
				shutdown_acksent,
				abort,
			};
		};

		struct IncomingResult {
			enum E {
				success,
				needsAck,
				failed,
			};
		};

		struct NotificationData {
			NotificationData* next;
			Notification::E notify;
			Handle h;
			Packet p;
			int n;
		};

		struct PacketData {
			Packet p;
			uint32_t n;
		};

		struct ConnectionData {
			std::deque<PacketData, mnet::allocator<PacketData>> pendingAck;
			int64_t retransmitTimeout;
			uint32_t bytesReceived;
			uint32_t outgoingSequence;
			uint32_t retransmits;
			bool disableAcks;
			bool receivedShutdown;
			bool receivedShutdownAck;
		};

		static void initializeConnectionData(ConnectionData* cd) {
			cd->retransmitTimeout = 0;
			cd->bytesReceived = 0;
			cd->outgoingSequence = 0;
			cd->retransmits = 0;
			cd->disableAcks = false;
			cd->receivedShutdown = false;
			cd->receivedShutdownAck = false;

			for (size_t ii = 0, nn = cd->pendingAck.size(); ii != nn; ++ii) {
				packetRelease(&cd->pendingAck[ii].p);
			}
			cd->pendingAck.clear();
		}

		struct Connection {
			Handle handle;
			ConnectionState::E state;
			uint8_t incomingTag[3];
			uint8_t outgoingTag[3];
			uint8_t buffer[65535];
			ConnectionData data;
			bool closeCalled;

			// state specific data
			union {
				struct {
					Packet  initPkt;
					int64_t resendTimeout;
					int     retries;
				} connecting;

				struct {
					Packet ackPkt;
					int64_t resendTimeout;
					int retries;
					int nackPkt;
				} connect_sendack;

				struct {
					int64_t abortTimeout;
					int64_t retransmitTimeout;
					uint32_t currentAck;
					int retries;
				} shutdown;
			};
		};

		struct ConnectionList {
			uint16_t* free;
			int nfree;
			Connection* all;
			int max;
		};

		struct Context {
			ConnectionList* connections;
			std::deque<NotificationData, mnet::allocator<NotificationData>> notifies;
		};
	} // namespace `anonymous'

	static Context* g_ctx;

	static void queueNotification(Handle h, Notification::E notify) {
		NotificationData nd;
		nd.h = h;
		nd.notify = notify;
		nd.next = NULL;
		g_ctx->notifies.push_back(nd);
	}

	static void closeConnection(Connection* c) {
		switch (c->state) {
		case ConnectionState::connecting:
			{
				packetRelease(&c->connecting.initPkt);
			} break;

		case ConnectionState::connect_sendecho:
			{
				packetRelease(&c->connect_sendack.ackPkt);
			} break;
		}

		if (c->state != ConnectionState::closed) {
			platform::socketClose(c->handle);
		}

		initializeConnectionData(&c->data);
		c->state = ConnectionState::closed;
	}

	// flush all pending-ACK data up through the specified incoming ACK
	static bool flushAckedData(Connection* c, uint32_t ack) {
		uint32_t remainingBytesToAck = ack - c->data.outgoingSequence;
		while (remainingBytesToAck > 0) {
			// if we don't have a pending packet to ACK, then we've
			// encountered a protocol error. The peer has acked more
			// data than we've sent
			if (c->data.pendingAck.empty()) {
				return false;
			}

			// ack bytes from first packet
			uint32_t bytesToAck = remainingBytesToAck;
			if (bytesToAck > c->data.pendingAck[0].n) {
				bytesToAck = c->data.pendingAck[0].n;
			}

			c->data.pendingAck[0].p.data += bytesToAck;
			c->data.pendingAck[0].n -= bytesToAck;

			// acked entire packet?
			if (c->data.pendingAck[0].n == 0) {
				packetRelease(&c->data.pendingAck[0].p);
				c->data.pendingAck.pop_front();
			}

			remainingBytesToAck -= bytesToAck;
			c->data.outgoingSequence += bytesToAck;
		}

		return true;
	}

	static IncomingResult::E updateIncoming(Connection* c, int n) {
		bool needsAck = false;

		// If we have new data, process that now
		if (n > 0) {
			switch (c->buffer[0]) {
			case Protocol::data:
				{
					if (n < 14) {
						return IncomingResult::failed;
					}

					uint16_t clientBytes;
					memcpy(&clientBytes, &c->buffer[12], 2);
					if (n != 14+clientBytes) {
						return IncomingResult::failed;
					}

					if (clientBytes > 0) {
						needsAck = true;
					}

					// ACK data sent to peer
					uint32_t incomingAck;
					memcpy(&incomingAck, &c->buffer[8], 4);
					if (!flushAckedData(c, incomingAck)) {
						return IncomingResult::failed;
					}

					// skip out-of order packets
					uint32_t incomingSequence;
					memcpy(&incomingSequence, &c->buffer[4], 4);
					int64_t diff = incomingSequence - (c->data.bytesReceived+clientBytes);
					if (diff == 0) {
						// stop the T2-rtx timer
						c->data.retransmitTimeout = 0;
						c->data.bytesReceived = incomingSequence;

						// queue the packet for the client
						if (n > 14) {
							NotificationData nd;
							nd.notify = Notification::packet;
							nd.h = c->handle;
							nd.p = packetAlloc(n-14);
							nd.n = n-14;
							memcpy(nd.p.data, &c->buffer[14], n-14);
							g_ctx->notifies.push_back(nd);
						}
					}

				} break;

			case Protocol::shutdown:
				{
					if (n == 8) {
						c->data.receivedShutdown = true;
						uint32_t incomingAck;
						memcpy(&incomingAck, &c->buffer[4], 4);
						if (!flushAckedData(c, incomingAck)) {
							return IncomingResult::failed;
						}
					}
				} break;

			case Protocol::shutdown_ack:
				{
					c->data.receivedShutdownAck = true;
				} break;

			case Protocol::cookie_ack:
				{
					// ignore duplicates
				} break;

			case Protocol::abort:
				return IncomingResult::failed;

			default:
				return IncomingResult::failed;
			}
		}

		return needsAck ? IncomingResult::needsAck : IncomingResult::success;
	}

	// Build and send a DATA packet to the peer. This will acknowledge received
	// data as well as transmit application data to the peer.
	static bool updateOutgoing(Connection* c, int64_t now) {
		// fill header fields that are not dependent on packet data
		uint32_t n = 14;
		c->buffer[0] = Protocol::data;
		memcpy(&c->buffer[1], c->outgoingTag, 3);
		memcpy(&c->buffer[8], &c->data.bytesReceived, 4);

		static const uint32_t maxDatagramSize = 512;

		// append data until we have a full packet
		uint16_t clientBytes = 0;
		uint32_t remaining = maxDatagramSize - n;
		for (size_t ii = 0, nn = c->data.pendingAck.size(); ii != nn; ++ii) {
			if (remaining == 0) {
				break;
			}

			uint32_t dataToWrite = c->data.pendingAck[ii].n;
			if (dataToWrite > remaining) {
				dataToWrite = remaining;
			}

			memcpy(&c->buffer[n], c->data.pendingAck[ii].p.data, dataToWrite);
			clientBytes += static_cast<uint16_t>(dataToWrite);
			remaining -= dataToWrite;
			n += dataToWrite;
		}

		// fill header fields that are dependent on packet data
		const uint32_t seq = c->data.outgoingSequence + static_cast<uint32_t>(clientBytes);
		memcpy(&c->buffer[4], &seq, 4);
		memcpy(&c->buffer[12], &clientBytes, 2);

		// send the packet to the peer
		platform::Result::E result = platform::socketSend(c->handle, c->buffer, n);
		if (result == platform::Result::failed) {
			return false;
		}

		// (re)start the retransmit timer
		if (clientBytes > 0) {
			c->data.retransmitTimeout = platform::timeAddNS(now, c_rentransmitTimeout);
		}
		
		return true;
	}

	static bool processReliabilityLayer(Connection* c, int64_t now, int n) {

		const bool retransmitSignaled = (c->data.retransmitTimeout != 0 && now > c->data.retransmitTimeout);
		if (retransmitSignaled) {
			c->data.retransmitTimeout = 0;
		}

		// process incoming data
		IncomingResult::E incomingRes = updateIncoming(c, n);
		if (incomingRes == IncomingResult::failed) {
			return false;
		}

		bool sendData = !c->data.pendingAck.empty() && c->data.retransmitTimeout == 0;

		// update resend logic
		if (n > 0) {
			c->data.retransmits = 0;
		} else if (retransmitSignaled) {
			++c->data.retransmits;
			if (c->data.retransmits > c_maxSequentialRetransmits) {
				return false;
			}
		}

		if (!c->data.disableAcks) {
			if (sendData || incomingRes == IncomingResult::needsAck) {
				if (!updateOutgoing(c, now)) {
					return false;
				}
			}
		}

		return true;
	}

	static void update(Connection* c, int64_t now) {

		// check for incoming packets
		int n = sizeof(c->buffer);
		platform::Result::E result = platform::socketRecv(c->handle, c->buffer, &n);
		if (result == platform::Result::failed) {
			queueNotification(c->handle, Notification::lostConnection);
			closeConnection(c);
			return;
		} else if (result == platform::Result::wouldBlock || n < 3) {
			n = 0;
		}

		// verify incoming tag of packet
		if (n != 0 && c->state != ConnectionState::connecting && c->state != ConnectionState::connect_sendecho) {
			if (0 != memcmp(c->incomingTag, &c->buffer[1], 3)) {
				n = 0;
			}
		}

		switch (c->state) {
		case ConnectionState::connecting:
			{
				if (c->closeCalled) {
					closeConnection(c);
					queueNotification(c->handle, Notification::failedToConnect);
					return;
				}

				// did we get a COOKIE packet?
				if (n >= 8) {
					if (c->buffer[0] == Protocol::cookie && 0 == memcmp(c->outgoingTag, &c->buffer[4], 3) && c->buffer[7] == Version::Version1) {
						c->state = ConnectionState::connect_sendecho;
						packetRelease(&c->connecting.initPkt);
						c->connect_sendack.retries = 5+1;
						c->connect_sendack.resendTimeout = now-1;
						c->connect_sendack.nackPkt = n;
						c->connect_sendack.ackPkt= packetAlloc(n);
						memcpy(c->connect_sendack.ackPkt.data, c->buffer, n);
						memcpy(c->incomingTag, &c->buffer[1], 3);
						c->connect_sendack.ackPkt.data[0] = Protocol::cookie_echo;
						break;
					}
				}

				// still waiting?
				if (now < c->connecting.resendTimeout) {
					break;
				}

				// out of retries?
				if (c->connecting.retries == 0) {
					closeConnection(c);
					queueNotification(c->handle, Notification::failedToConnect);
					return;
				}

				// resend the INIT packet
				--c->connecting.retries;
				c->connecting.resendTimeout = platform::timeAddNS(now, c_connectTimeout);

				platform::Result::E result = platform::socketSend(c->handle, c->connecting.initPkt.data, 9);
				if (result == platform::Result::failed) {
					closeConnection(c);
					queueNotification(c->handle, Notification::failedToConnect);
					break;
				}

			} break;

		case ConnectionState::connect_sendecho:
			{
				if (c->closeCalled) {
					closeConnection(c);
					queueNotification(c->handle, Notification::failedToConnect);
					return;
				}

				// did we get a COOKIE-ECHO packet?
				if (n == 4) {
					if (c->buffer[0] == Protocol::cookie_ack && 0 == memcmp(c->incomingTag, &c->buffer[1], 3)) {
						c->state = ConnectionState::connected;
						packetRelease(&c->connect_sendack.ackPkt);
						queueNotification(c->handle, Notification::connected);
						break;
					}
				}

				// still waiting?
				if (now < c->connect_sendack.resendTimeout) {
					break;
				}

				// out of rerties?
				if (c->connect_sendack.retries == 0) {
					closeConnection(c);
					queueNotification(c->handle, Notification::failedToConnect);
					return;
				}

				// resend the COOKIE-ECHO packet
				--c->connect_sendack.retries;
				c->connect_sendack.resendTimeout = platform::timeAddNS(now, c_connectTimeout);

				platform::Result::E result = platform::socketSend(c->handle, c->connect_sendack.ackPkt.data, c->connect_sendack.nackPkt);
				if (result == platform::Result::failed) {
					closeConnection(c);
					queueNotification(c->handle, Notification::failedToConnect);
					break;
				}
			} break;

		case ConnectionState::connected:
			{
				if (!processReliabilityLayer(c, now, n)) {
					c->state = ConnectionState::abort;
					return;
				}

				// it's an error to receive a SHUTDOWN-ACK packet here
				if (c->data.receivedShutdownAck) {
					c->state = ConnectionState::abort;
					return;
				}

				if (c->closeCalled) {
					c->state = ConnectionState::shutdown_pending;
					c->shutdown.abortTimeout = platform::timeAddNS(now, c_shutdownTimeout);
					return;
				}

				if (c->data.receivedShutdown) {
					c->state = ConnectionState::shutdown_received;
					c->shutdown.abortTimeout = platform::timeAddNS(now, c_shutdownTimeout);
					return;
				}
			} break;

		case ConnectionState::shutdown_pending:
			{
				// if we've expsired the abort timer, bail out
				if (now > c->shutdown.abortTimeout) {
					c->state = ConnectionState::abort;
					return;
				}

				// has all data been acked?
				if (c->data.pendingAck.empty()) {
					c->state = ConnectionState::shutdown_sent;
					c->shutdown.currentAck = c->data.bytesReceived;
					c->shutdown.retransmitTimeout = platform::timeAddNS(now, c_shutdownPacketTimeout);
					c->shutdown.retries = 5+1;
					return;
				}

				// process client data
				if (!processReliabilityLayer(c, now, n)) {
					c->state = ConnectionState::abort;
					return;
				}

			} break;

		case ConnectionState::shutdown_received:
			{
				// if we've expired the abort timer, bail out
				if (now > c->shutdown.abortTimeout) {
					c->state = ConnectionState::abort;
					return;
				}

				// has all data been acknowledged
				if (c->data.pendingAck.size() == 0) {
					c->state = ConnectionState::shutdown_acksent;
					c->data.disableAcks = true;
					c->shutdown.retransmitTimeout = 0;
					c->shutdown.retries = 5+1;
					return;
				}

				// process client data
				if (!processReliabilityLayer(c, now, n)) {
					c->state = ConnectionState::abort;
					return;
				}

			} break;

		case ConnectionState::shutdown_sent:
			{
				// if we have a new outgoing sequence, (re)send SHUTDOWN packet
				uint32_t seq = c->data.bytesReceived;
				if (seq != c->shutdown.currentAck || now > c->shutdown.retransmitTimeout) {
					// process resends of teh SHUTDOWN packet
					if (seq == c->shutdown.currentAck) {
						--c->shutdown.retries;
						if (c->shutdown.retries == 0) {
							c->state = ConnectionState::abort;
							return;
						}
					}

					// build SHUTDOWN packet
					char buffer[8];
					buffer[0] = Protocol::shutdown;
					memcpy(&buffer[1], c->outgoingTag, 3);
					memcpy(&buffer[4], &c->data.bytesReceived, 4);

					platform::Result::E result = platform::socketSend(c->handle, buffer, 8);
					if (result == platform::Result::failed) {
						c->state = ConnectionState::abort;
						return;
					}

					// start T2-rtx to resent SHUTDOWN
					c->shutdown.retransmitTimeout = platform::timeAddNS(now, c_shutdownPacketTimeout);
				}

				// on SHUTDOWN, transition to shutdown_acksent
				if (c->data.receivedShutdown) {
					c->state = ConnectionState::shutdown_acksent;
					c->data.disableAcks = true;
					c->shutdown.retransmitTimeout = 0;
					c->shutdown.retries = 5+1;
					return;
				}

				// on SHUTDOWN-ACK, transition to closed
				if (c->data.receivedShutdownAck) {
					// send SHUTDOWN-COMPLETE
					char buffer[4];
					buffer[0] = Protocol::shutdown_complete;
					memcpy(&buffer[1], c->outgoingTag, 3);

					platform::socketSend(c->handle, buffer, 4);
					closeConnection(c);
					queueNotification(c->handle, Notification::closed);
					return;
				}

				if (now > c->shutdown.abortTimeout) {
					c->state = ConnectionState::abort;
					return;
				}

				// process incoming data
				if (!processReliabilityLayer(c, now, n)) {
					c->state = ConnectionState::abort;
					return;
				}

			} break;

		case ConnectionState::shutdown_acksent:
			{
				// (re)transmit SHUTDOWN-ACK packet
				if (now > c->shutdown.retransmitTimeout) {
					--c->shutdown.retries;
					if (c->shutdown.retries == 0) {
						c->state = ConnectionState::abort;
						return;
					}

					// build SHUTDOWN-ACK packet
					char buffer[4];
					buffer[0] = Protocol::shutdown_ack;
					memcpy(&buffer[1], c->outgoingTag, 3);

					platform::Result::E result = platform::socketSend(c->handle, buffer, 4);
					if (result == platform::Result::failed) {
						c->state = ConnectionState::abort;
						return;
					}

					c->shutdown.retransmitTimeout = platform::timeAddNS(now, c_shutdownPacketTimeout);
				}

				// has the abort timer been signaled?
				if (now > c->shutdown.abortTimeout) {
					c->state = ConnectionState::abort;
					return;
				}

				// have we received a SHUTDOWN-COMPLETE packet?
				if (n > 0 && c->buffer[0] == Protocol::shutdown_complete) {
					closeConnection(c);
					queueNotification(c->handle, Notification::closed);
					return;
				}

			} break;

		case ConnectionState::abort:
			{
				// send ABORT
				char buffer[4];
				buffer[0] = Protocol::abort;
				memcpy(&buffer[1], c->outgoingTag, 3);
				platform::socketSend(c->handle, buffer, 4);

				closeConnection(c);
				queueNotification(c->handle, Notification::lostConnection);
			} break;
		};
	}

	// update the state of all connections
	static void updateConnections(int64_t now) {
		for (int ii = 0, max = g_ctx->connections->max; ii < max; ++ii) {
			Connection* c = &g_ctx->connections->all[ii];
			if (c->state != ConnectionState::closed) {
				update(c, now);
			}
		}
	}

	const Handle invalidHandle = {0xffff};
	const Handle loopbackHandle = {0xfffe};

	bool initialize(int maxConnections, const Allocator* alloc) {
		if (maxConnections > 0xffff) return false;

		if (alloc) setAllocator(alloc);

		if (!platform::startup(maxConnections, getAllocator())) {
			return false;
		}

		// allocate connections
		void* contextBlock = mnet::alloc(sizeof(Context)+sizeof(ConnectionList)+maxConnections*(sizeof(Connection)+sizeof(uint16_t)));
		g_ctx = new(contextBlock) Context;
		g_ctx->connections = reinterpret_cast<ConnectionList*>(g_ctx+1);
		g_ctx->connections->all = reinterpret_cast<Connection*>(g_ctx->connections + 1);
		g_ctx->connections->max = maxConnections;
		g_ctx->connections->free = reinterpret_cast<uint16_t*>(g_ctx->connections->all + maxConnections);
		g_ctx->connections->nfree = maxConnections;
		for (uint16_t ii = 0; ii < (uint16_t)maxConnections; ++ii) {
			g_ctx->connections->free[ii] = ii;
			new(&g_ctx->connections->all[ii]) Connection;
			g_ctx->connections->all[ii].handle.index = (uint16_t)ii;
			g_ctx->connections->all[ii].state = ConnectionState::closed;
		}

		return true;
	}

	void shutdown() {
		for (size_t ii = 0, nn = g_ctx->notifies.size(); ii != nn; ++ii) {
			if (g_ctx->notifies[ii].notify == Notification::packet) {
				packetRelease(&g_ctx->notifies[ii].p);
			}
		}
		g_ctx->notifies.clear();

		for (int ii = 0, nn = g_ctx->connections->max; ii != nn; ++ii) {
			closeConnection(&g_ctx->connections->all[ii]);
			g_ctx->connections->all[ii].~Connection();
		}

		platform::shutdown();

		void* contextBlock = g_ctx;
		mnet::free(contextBlock);
		g_ctx = 0;
		setAllocator(0);
	}

	bool connect(Handle* h, uint32_t ip, uint16_t port) {
		// do we have an available handle?
		if (g_ctx->connections->nfree == 0) return false;

		// allocate a connection
		Handle handle = {g_ctx->connections->free[g_ctx->connections->nfree-1]};

		const int64_t now = platform::timestamp();

		// create a socket for the connection
		platform::Result::E result = platform::socketCreate(handle, ip, port);
		if (result != platform::Result::success) {
			return false;
		}

		Connection* c = &g_ctx->connections->all[handle.index];

		// generate incoing connection tag
		result = platform::fillRand(c->outgoingTag, sizeof(c->outgoingTag));
		if (result != platform::Result::success) {
			platform::socketClose(handle);
			return false;
		}

		// generate the init packet
		Packet p = packetAlloc(9);
		p.data[0] = Protocol::init;
		p.data[1] = 'M';
		p.data[2] = 'N';
		p.data[3] = 'E';
		p.data[4] = 'T';
		p.data[5] = Version::Version1;
		p.data[6] = c->outgoingTag[0];
		p.data[7] = c->outgoingTag[1];
		p.data[8] = c->outgoingTag[2];

		// start the connection process
		initializeConnectionData(&c->data);
		c->state = ConnectionState::connecting;
		c->connecting.resendTimeout = now - 1;
		c->connecting.retries = 5+1;
		c->connecting.initPkt = p;
		c->closeCalled = false;
		update(c, now);

		// commit the handle allocation
		--g_ctx->connections->nfree;
		*h = handle;
		return true;
	}

	void disconnect(Handle* h) {
		const int64_t now = platform::timestamp();
		Connection* c = &g_ctx->connections->all[h->index];
		c->closeCalled = true;
		update(c, now);

		*h = invalidHandle;
	}

	void send(Handle h, Packet p, int size) {
		const int64_t now = platform::timestamp();

		// handle loopback packets
		if (h.index == loopbackHandle.index) {
			NotificationData nd;
			nd.notify = Notification::packet;
			nd.h = h;
			nd.p = p;
			nd.n = size;
			packetAddref(p);
			g_ctx->notifies.push_back(nd);
			return;
		}

		// drop packet if we are in states
		Connection* c = &g_ctx->connections->all[h.index];
		const bool allowSend = (c->state == ConnectionState::connected || c->state == ConnectionState::connecting || c->state == ConnectionState::connect_sendecho);
		if (allowSend) {

			PacketData pd;
			pd.p = p;
			pd.n = static_cast<uint32_t>(size);
			packetAddref(p);
			c->data.pendingAck.push_back(pd);
		}

		updateConnections(now);
	}

	Notification::E recv(Handle* h, Packet* p, int* size) {
		const int64_t now = platform::timestamp();
		updateConnections(now);

		if (!g_ctx->notifies.empty()) {
			NotificationData nd = g_ctx->notifies.front();
			g_ctx->notifies.pop_front();

			*h = nd.h;
			*p = nd.p; // transfer ownership (if any) to caller
			*size = nd.n;
			return nd.notify;
		}
		return Notification::none;
	}

	uint32_t parseIPv4(const char* ip) {
		uint32_t a, b, c, d;
		int n = sscanf_s(ip, "%d.%d.%d.%d", &a, &b, &c, &d);
		if (n == 4) {
			return (a << 24) | (b << 16) | (c << 8) | d;
		}

		return 0;
	}

} // namespace mnet
