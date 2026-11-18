#include "http.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <thread>
#include "rt.h"

namespace glm {

namespace {
const char* reason(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
    }
    return "Status";
}

bool write_all(int fd, const char* p, size_t n) {
    while (n) {
        ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) { if (w < 0 && errno == EINTR) continue; return false; }
        p += w;
        n -= w;
    }
    return true;
}

// read one request: headers up to the blank line, then Content-Length bytes of body
bool read_request(int fd, HttpRequest& r) {
    std::string buf;
    char tmp[65536];
    size_t hend;
    while ((hend = buf.find("\r\n\r\n")) == std::string::npos) {
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf.append(tmp, n);
        if (buf.size() > (64u << 20)) return false;
    }
    std::string head = buf.substr(0, hend);
    size_t le = head.find("\r\n");
    std::string line = head.substr(0, le);
    size_t s1 = line.find(' '), s2 = line.find(' ', s1 + 1);
    if (s1 == std::string::npos) return false;
    r.method = line.substr(0, s1);
    r.path = line.substr(s1 + 1, s2 == std::string::npos ? std::string::npos : s2 - s1 - 1);
    size_t p = le == std::string::npos ? head.size() : le + 2;
    while (p < head.size()) {
        size_t e = head.find("\r\n", p);
        if (e == std::string::npos) e = head.size();
        std::string h = head.substr(p, e - p);
        size_t c = h.find(':');
        if (c != std::string::npos) {
            std::string k = h.substr(0, c), v = h.substr(c + 1);
            std::transform(k.begin(), k.end(), k.begin(), ::tolower);
            v.erase(0, v.find_first_not_of(" \t"));
            r.headers[k] = v;
        }
        p = e + 2;
    }
    size_t len = r.headers.count("content-length") ? std::stoul(r.headers["content-length"]) : 0;
    r.body = buf.substr(hend + 4);
    while (r.body.size() < len) {
        ssize_t n = recv(fd, tmp, std::min(sizeof tmp, len - r.body.size()), 0);
        if (n <= 0) return false;
        r.body.append(tmp, n);
    }
    r.body.resize(std::min(r.body.size(), len));
    return true;
}
}  // namespace

void HttpConn::send(int code, const std::string& type, const std::string& body) {
    std::string h = "HTTP/1.0 " + std::to_string(code) + " " + reason(code) + "\r\nContent-Type: " + type +
                    "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    write_all(fd_, h.data(), h.size()) && write_all(fd_, body.data(), body.size());
}

void HttpConn::start_stream() {
    static const char h[] = "HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n";
    write_all(fd_, h, sizeof h - 1);
}

bool HttpConn::write(const std::string& data) { return write_all(fd_, data.data(), data.size()); }

bool socket_peer_gone(int fd) {
    pollfd p{fd, POLLIN | POLLRDHUP, 0};
    if (poll(&p, 1, 0) <= 0) return false;
    if (p.revents & (POLLHUP | POLLERR | POLLRDHUP | POLLNVAL)) return true;
    char c;
    return recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
}

bool HttpConn::client_gone() const { return socket_peer_gone(fd_); }

HttpServer::HttpServer(const std::string& host, int port) {
    lfd_ = socket(AF_INET, SOCK_STREAM, 0);
    GLM_CHECK(lfd_ >= 0, "socket failed");
    int one = 1;
    setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    GLM_CHECK(inet_pton(AF_INET, host == "localhost" ? "127.0.0.1" : host.c_str(), &a.sin_addr) == 1,
              "bad host %s", host.c_str());
    GLM_CHECK(bind(lfd_, (sockaddr*)&a, sizeof a) == 0, "cannot bind %s:%d (%s)", host.c_str(), port, strerror(errno));
    GLM_CHECK(listen(lfd_, 64) == 0, "listen failed");
}

HttpServer::~HttpServer() {
    if (lfd_ >= 0) close(lfd_);
}

void HttpServer::serve(Handler h) {
    std::thread([this, h] {
        for (;;) {
            sockaddr_in pa{};
            socklen_t pl = sizeof pa;
            int fd = accept(lfd_, (sockaddr*)&pa, &pl);
            if (fd < 0) { if (errno == EINTR) continue; break; }
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            char ip[64];
            inet_ntop(AF_INET, &pa.sin_addr, ip, sizeof ip);
            std::string peer = ip;
            std::thread([fd, h, peer] {
                HttpRequest r;
                r.peer = peer;
                if (read_request(fd, r)) {
                    HttpConn c(fd);
                    try {
                        h(r, c);
                    } catch (const std::exception& e) {
                        std::string b = std::string("{\"error\":{\"message\":\"") + e.what() + "\",\"code\":500}}";
                        c.send(500, "application/json", b);
                    }
                }
                close(fd);
            }).detach();
        }
    }).detach();
}

}  // namespace glm
