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

#include <stdio.h>
#include <mnet/mnet.h>
#include <mnet/platform.h>
#include <Windows.h>

int main() {

	if (!mnet::initialize(5)) {
		fprintf(stderr, "Failed to initialize mnet\n");
		return -1;
	}

	uint32_t ip = mnet::parseIPv4("45.33.55.92");
	uint16_t port = 26999;

	mnet::Handle h;
	if (!mnet::connect(&h, ip, port)) {
		fprintf(stderr, "Failed to connect to remote host\n");
		return -1;
	}

	char buffer[] = "GET /status HTTP/1.1\r\n"
		"Content-Size: 0\r\n"
		"Host: 3535388524\r\n"
		"\r\n"
		;
	mnet::Packet p = mnet::packetAlloc(sizeof(buffer)-1);
	memcpy(p.data, buffer, sizeof(buffer)-1);
	mnet::send(h, p, sizeof(buffer)-1);

	bool running = true;
	for (;running ;) {
		mnet::Handle h;
		mnet::Packet p;
		int n;
		mnet::Notification::E notify = mnet::recv(&h, &p, &n);
		switch (notify) {
		case mnet::Notification::none:
			Sleep(100);
			break;
		case mnet::Notification::failedToConnect:
			fprintf(stderr, "Failed to connect\n");
			running = false;
			break;
		case mnet::Notification::lostConnection:
			fprintf(stderr, "Lost connection\n");
			running = false;
			break;
		case mnet::Notification::connected:
			printf("Connected to remote host\n");
			break;
		case mnet::Notification::packet:
			printf("DAT: %.*s\n", n, p.data);
			mnet::packetRelease(&p);
			break;
		case mnet::Notification::closed:
			printf("closed\n");
			running = false;
			break;
		default:
			printf("Got notification: %d\n", notify);
			break;
		}
	}

	mnet::disconnect(&h);
	mnet::shutdown();
}
