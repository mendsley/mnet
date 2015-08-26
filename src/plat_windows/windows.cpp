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

#if !defined(MNET_DISABLE_BUILTIN_PLATFORMS)

#include <mnet/platform.h>
#include <mnet/mnet.h>
#include <mnet/memory.h>

#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <stdint.h>
#include <Windows.h>
#include <WinSock2.h>
#include <wincrypt.h>

#if !defined(MNET_AUTO_LINK_LIBRARIES)
#pragma comment(lib, "ws2_32.lib")
#endif // !defined(MNET_AUTO_LINK_LIBRARIES)

namespace mnet {
	namespace platform {

		namespace {

			struct Socket {
				SOCKET s;
				int nraddr;
				sockaddr_storage raddr;
			};

			struct Context {
				int64_t performanceFreq;
				const Allocator* alloc;
				HCRYPTPROV crypto;

				Socket* socks;
			};

		} // namespace `anonymous'

		static Context g_platform;

		bool startup(int maxConnections, const Allocator* alloc) {
			LARGE_INTEGER freq;
			QueryPerformanceFrequency(&freq);
			g_platform.performanceFreq = freq.QuadPart;

			// create a context for random generation
			if (!CryptAcquireContext(&g_platform.crypto, NULL, NULL, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
				return false;
			}
			
			// initialize winsock
			WSADATA wsad;
			if (0 != WSAStartup(MAKEWORD(2, 0), &wsad)) {
				CryptReleaseContext(g_platform.crypto, 0);
				return false;
			}

			// allocate sockets
			g_platform.alloc = alloc;
			g_platform.socks = static_cast<Socket*>(alloc->realloc(0, sizeof(Socket)*maxConnections));
			for (int ii = 0; ii < maxConnections; ++ii) {
				g_platform.socks[ii].s = INVALID_SOCKET;
			}
			return true;
		}

		void shutdown() {
			g_platform.alloc->realloc(g_platform.socks, 0);
			WSACleanup();
			CryptReleaseContext(g_platform.crypto, 0);

			g_platform.socks = 0;
			g_platform.crypto = NULL;
			g_platform.alloc = NULL;
		}

		int64_t timestamp() {
			LARGE_INTEGER pc;
			QueryPerformanceCounter(&pc);
			return pc.QuadPart;
		}

		int64_t timeAddNS(int64_t ts, int64_t ns) {
			return ts + ns*g_platform.performanceFreq/1000000000;
		}

		uint32_t atomicInc(volatile uint32_t* v) {
			return InterlockedIncrement(reinterpret_cast<volatile unsigned int*>(v));
		}

		uint32_t atomicDec(volatile uint32_t* v) {
			return InterlockedDecrement(reinterpret_cast<volatile unsigned int*>(v));
		}

		Result::E fillRand(uint8_t* p, int n) {
			if (!CryptGenRandom(g_platform.crypto, n, p)) {
				return Result::failed;
			}

			return Result::success;
		}

		Result::E socketCreate(Handle h, uint32_t ip, uint16_t port) {
			Socket* s = &g_platform.socks[h.index];
			s->s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
			if (s->s == INVALID_SOCKET) {
				return Result::failed;
			}

			// set the socket to non blocking
			u_long opt = 1;
			if (0 != ioctlsocket(s->s, FIONBIO, &opt)) {
				closesocket(s->s);
				return Result::failed;
			}

			// configure remote address
			memset(&s->raddr, 0, sizeof(s->raddr));
			s->nraddr = sizeof(sockaddr_in);
			sockaddr_in* inaddr = reinterpret_cast<sockaddr_in*>(&s->raddr);
			inaddr->sin_family      = AF_INET;
			inaddr->sin_addr.s_addr = htonl(ip);
			inaddr->sin_port        = htons(port);

			// bind to local address
			sockaddr_in laddr;
			memset(&laddr, 0, sizeof(laddr));
			laddr.sin_family      = AF_INET;
			laddr.sin_addr.s_addr = INADDR_ANY;
			laddr.sin_port        = 0;
			if (0 != bind(s->s, reinterpret_cast<const sockaddr*>(&laddr), sizeof(laddr))) {
				closesocket(s->s);
				return Result::failed;
			}

			return Result::success;
		}

		void socketClose(Handle h) {
			Socket* s = &g_platform.socks[h.index];
			closesocket(s->s);
			s->s = INVALID_SOCKET;
		}

		Result::E socketSend(Handle h, const void* p, int n) {
			Socket* s = &g_platform.socks[h.index];
			int sent = sendto(s->s, static_cast<const char*>(p), n, 0, reinterpret_cast<const sockaddr*>(&s->raddr), s->nraddr);
			if (sent == n) {
				return Result::success;
			}
			if (sent < 0) {
				int error = WSAGetLastError();
				if (error == WSAEWOULDBLOCK) {
					return Result::wouldBlock;
				}

				return Result::failed;
			}

			// partial sends are failures
			return Result::failed;
		}

		Result::E socketRecv(Handle h, void* p, int* n) {
			Socket* s = &g_platform.socks[h.index];

			sockaddr_storage raddr;
			int nraddr = sizeof(raddr);
			int recvd = recvfrom(s->s, reinterpret_cast<char*>(p), *n, 0, reinterpret_cast<sockaddr*>(&raddr), &nraddr);
			if (recvd <= 0) {
				int error = WSAGetLastError();
				if (error == WSAEWOULDBLOCK) {
					return Result::wouldBlock;
				}

				return Result::failed;
			}

			// does this match our remote host?
			if (nraddr != s->nraddr || 0 != memcmp(&raddr, &s->raddr, nraddr)) {
				return Result::wouldBlock;
			}

			*n = recvd;
			return Result::success;
		}

	} // namespace platform
} // namespace mnet

#endif // !defined(MNET_BUILTIN_PLATFORMS)
