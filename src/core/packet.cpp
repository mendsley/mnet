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

#include "core/memory.h"

namespace mnet {

	Packet packetAlloc(int size) {
		void* block = mnet::alloc(sizeof(uint32_t)+size);

		Packet p;
		p.refcnt = static_cast<uint32_t*>(block);
		p.data   = static_cast<uint8_t*>(block) + sizeof(uint32_t);
		return p;
	}

	void packetRelease(Packet* p) {
		if (0 == platform::atomicDec(reinterpret_cast<unsigned int*>(p->refcnt))) {
			void* block = static_cast<void*>(p->refcnt);
			mnet::free(block);
		}

		p->refcnt = 0;
		p->data   = 0;
	}

	void packetAddref(Packet p) {
		platform::atomicInc(reinterpret_cast<unsigned int*>(p.refcnt));
	}

} // namespace mnet
