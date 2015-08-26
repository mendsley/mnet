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

#ifndef HEADERGUARD__MNET__MNET_H
#define HEADERGUARD__MNET__MNET_H

#include <stdint.h>

namespace mnet {

	struct Allocator;

	struct Packet {
		uint32_t* refcnt;
		uint8_t*  data;
	};

	struct Notification {
		enum E {
			none, // nothing to report
			packet,
			connected,
			failedToConnect,
			lostConnection,
			closed,
		};
	};

	Packet packetAlloc(int size);
	void packetRelease(Packet* p);
	void packetAddref(Packet p);

	struct Handle { uint16_t index; };

	extern const Handle invalidHandle;
	extern const Handle loopbackHandle;
	static inline bool isValid(Handle h) { return h.index != invalidHandle.index; }

	bool initialize(int max_connections, const Allocator* alloc = 0);
	void shutdown();

	bool connect(Handle* h, uint32_t ip, uint16_t port);
	void disconnect(Handle* h);

	// queues data to be sent on the connection
	void send(Handle h, Packet p, int size);

	// Dequeue a single notification and return.
	//
	// If no notification is available, returns `Notification::none`. If
	// `Notification::packet` is returned, then `p` and `size` are filled in. `p`
	// must be released with `mnet::packetRelease`.
	Notification::E recv(Handle *h, Packet* p, int* size);

	uint32_t parseIPv4(const char* ip);

} // namespace mnet

#endif // HEADERGUARD__MNET__MNET_H
