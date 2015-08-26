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

#ifndef HEADERGUARD__MNET__PLATFORM_H
#define HEADERGUARD__MNET__PLATFORM_H

#include <stdint.h>

namespace mnet {
	struct Allocator;
	struct Handle;
	struct Packet;

	namespace platform {

		struct Result {
			enum E {
				success,    // operation succeeded.
				failed,     // operation failed.
				wouldBlock, // operation would have blocket, but may succeed if called again
			};
		};


		bool startup(int maxConnections, const Allocator* alloc);
		void shutdown();

		// Get the current time in nanoseconds.
		//
		// Timestamps are only used relative to another, so they do not need to be
		// anchored to a specific epoch.
		int64_t timestamp();

		// Adds `ns` nanoseconds to a timestamp `ts` and returns the new timestamp
		int64_t timeAddNS(int64_t ts, int64_t ns);


		uint32_t atomicInc(volatile uint32_t* v);
		uint32_t atomicDec(volatile uint32_t* v);


		// Fill a buffer with random data
		Result::E fillRand(uint8_t* p, int n);


		// Create a non-blocking UDP socket.
		//
		// All sends from this socket should be addressed to `ip`:`port`
		Result::E socketCreate(Handle h, uint32_t ip, uint16_t port);
		void socketClose(Handle h);
		Result::E socketSend(Handle h, const void* p, int n);

		// Read packet data from the socket
		//
		// Must drop packets that originate from a host other than the on specified
		// via `socketCreate`
		Result::E socketRecv(Handle h, void* p, int* n);

	} // namespace platform
} // namesapce mnet

#endif // HEADERGUARD__MNET__PLATFORM_H
