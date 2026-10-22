#pragma once

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

struct NetworkData
{
    SOCKET recv_sock = INVALID_SOCKET;
    SOCKET send_sock = INVALID_SOCKET;
    sockaddr_in send_addr;

    NetworkData(const std::string& ip, int recv_port, int send_port)
    {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);

        // 1. Create receiving UDP socket and bind to recv_port
        recv_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        sockaddr_in recv_addr{};
        recv_addr.sin_family = AF_INET;
        recv_addr.sin_port = htons(recv_port);
        recv_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // localhost only
        if (bind(recv_sock, (sockaddr*)&recv_addr, sizeof(recv_addr)) == SOCKET_ERROR)
            fprintf(stderr, "[Network] bind to port %d failed (error %d). Is another instance running?\n",
                    recv_port, WSAGetLastError());

        // Make recv_sock non-blocking so rendering loop never freezes
        u_long mode = 1;
        ioctlsocket(recv_sock, FIONBIO, &mode);

        // 2. Create sending UDP socket and destination address
        send_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        memset(&send_addr, 0, sizeof(send_addr));
        send_addr.sin_family = AF_INET;
        send_addr.sin_port = htons(send_port);
        send_addr.sin_addr.s_addr = inet_addr(ip.c_str());
    }

    ~NetworkData()
    {
        if (recv_sock != INVALID_SOCKET) closesocket(recv_sock);
        if (send_sock != INVALID_SOCKET) closesocket(send_sock);
        WSACleanup();
    }

    void send_state(const std::vector<float>& state)
    {
        sendto(send_sock, (const char*)state.data(), (int)(state.size() * sizeof(float)), 0,
               (sockaddr*)&send_addr, sizeof(send_addr));
    }

    // Blocks up to timeout_ms until a packet is readable. Lets the main loop idle without spinning
    // while still waking instantly when Python's next command arrives.
    bool wait_readable(int timeout_ms)
    {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(recv_sock, &fds);
        timeval tv{0, timeout_ms * 1000};
        return select(0, &fds, nullptr, nullptr, &tv) > 0;
    }

    bool receive_actions(std::vector<float>& actions)
    {
        float buffer[4096];
        int bytes = recvfrom(recv_sock, (char*)buffer, sizeof(buffer), 0, nullptr, nullptr);
        if (bytes > 0)
        {
            int num_floats = bytes / sizeof(float);
            actions.assign(buffer, buffer + num_floats);
            return true;
        }
        return false;
    }
};
