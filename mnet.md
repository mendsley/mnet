Packet types
============



INIT
====
0        1                                5        6                        9
0        8                                40       48                       72
+--------+--------------------------------+--------+------------------------+
|   1    |          ProtocolMagic         | Version|      ConnectionTag     |
+--------+--------------------------------+--------+------------------------+

Initiates a new unique connected session.

ProtocolMagic:
   Protocol identifier: 'M' 'N' 'E' 'T'
Version:
	Protocol version. This document describes version 1
ConnectionTag:
	Random 24bit number generated by the connecting peer that establishes
	a unique session. Must be included in all subsequent packets from teh
	server


COOKIE
======
0        1                        4                        7        8                                12                      32
0        8                        32                       48       56                               96                      256
+--------+------------------------+------------------------+--------+--------------------------------+-------- ~~ -----------+
|   2    |     ConnectionTag      |    PeerConnectionTag   | Version|            Expiration          |        HMAC           |
+--------+------------------------+------------------------+--------+--------------------------------+-------- ~~ -----------+

Confirms receipt of an INIT packet.

ConnectionTag:
	Listener->Connector tag. Random 24bit number generated by the listener
	that establishes a unique session on the connecting peer.
PeerConnectionTag:
	ConnectionTag sent from the connecting peer via an INIT packet
HMAC:
	SHA1 hmac of Expiration|PeerIP|LocalPort|ConnectionTag|PeerConnectionTag

COOKIE-ECHO
===========
0        1                                                                      32
0        8                                                                      256
+--------+--------------------------------- ~~ ---------------------------------+
|   3    |                         CookiePacketPayload                          |
+--------+--------------------------------- ~~ ---------------------------------+

Confirms ownership of the IP/Port tuple that originated the
initial INIT packet. DATA packets may now be sent from the
connecting peer.

CookiePacketPayload:
	Byte-by-byte copy of the COOKIE message received from the
	listener's COOKIE packet


COOKIE-ACK
==========
0        1                        4
0        8                        32
+--------+------------------------+
|   4    |   PeerConnectionTag    |
+--------+------------------------+

Confirms establishment of a connected session. DATA packets may now
be sent from the listener.

PeerConnectionTag:
	Connection tag received in the initial INIT message


DATA
====
0        1                        4                                8                                12               14
0        8                        32                               64                               96               112
+--------+------------------------+--------------------------------+--------------------------------+----------------+-------- ~~ ------+
|   0    |   PeerConnectionTag    |        OutgoingSequence        |       AcknowledgedBytes        |   DataLength   | Application Data |
+--------+------------------------+--------------------------------+--------------------------------+----------------+-------- ~~ ------+

Transports application data to the peer.

PeerConnectionTag:
	Connection tag received in the initial INIT message
OutgoingSequence:
	Total number of bytes (including the payload of this message) that
	have been sent to the peer so far
AcknowlegedBytes:
	Total number of bytes that have been recieved from the remote peer
DataLength:
	Length of the fragment's application data

SHUTDOWN
========
0        1                        4                                8
0        8                        32                               64
+--------+------------------------+--------------------------------+
|   5    |   PeerConnectionTag    |       AcknowledgedBytes        |
+--------+------------------------+--------------------------------+

Starts a graceful shutdown of a peer connection. Local host can
no longer send new DATA packets.

PeerConnectionTag:
	Connection tag received in the initial INIT message
AcknowlegedBytes:
	Total number of bytes that have been recieved from the remote peer

SHUTDOWN-ACK
============
0        1                        4
0        8                        32
+--------+------------------------+
|   5    |   PeerConnectionTag    |
+--------+------------------------+

Acknowleges receipt of a SHUTDOWN packet

PeerConnectionTag:
	Connection tag received in the initial INIT message

SHUTDOWN-COMPLETE
=================
0        1                        4
0        8                        32
+--------+------------------------+
|   5    |   PeerConnectionTag    |
+--------+------------------------+

Acknowleges receipt of a SHUTDOWN-ACK packet

PeerConnectionTag:
	Connection tag received in the initial INIT message

ABORT
=====
0        1                        4
0        8                        32
+--------+------------------------+
|   8    |    PeerConnectionTag   |
+--------+------------------------+

Aborts a session. Receiving peer should assume no further packet
will be delivered to the peeer's application layer.

PeerConnectionTag:
	Connection tag received in the initial INIT message