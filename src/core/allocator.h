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

#ifndef HEADERGUARD__MNET__CORE__ALLOCATOR_H
#define HEADERGUARD__MNET__CORE__ALLOCATOR_H

#include <new>
#include <stddef.h>
#include "core/memory.h"

namespace mnet {

	template<typename T>
	struct allocator {
		typedef T value_type;
		typedef T* pointer;
		typedef size_t size_type;
		
		template<typename U>
		struct rebind {
			typedef allocator<U> other;
		};

		inline allocator() {}
		explicit inline allocator(const allocator&) {}
		template<typename U>
		explicit inline allocator(const allocator<U>&) {}

		static inline pointer address(T& r) { return &r; }
		static inline const pointer address(const T& r) { return &r; }
		static inline pointer allocate(size_type count, const void* /*hint*/ = 0) { return static_cast<pointer>(mnet::alloc(count * sizeof(T))); }
		static inline void deallocate(pointer p, size_type) { mnet::free(p); }

		static inline void construct(pointer p, const T& t) { new(p) T(t); }
		static inline void destroy(pointer p) { p = p; p->~T(); }

		static inline size_t max_size() { return ~static_cast<size_t>(0); }
	};

	template<typename T, typename U>
	static inline bool operator==(const allocator<T>&, const allocator<U>&) { return true; }

	template<typename T, typename U>
	static inline bool operator!=(const allocator<T>&, const allocator<U>&) { return false; }
}

#endif // HEADERGUARD__MNET__CORE__ALLOCATOR_H
