#pragma once
// A small HTTP/1.0 server: one thread per connection, the request body read by Content-Length, responses closed by
// the connection (streams carry no length), server-sent events for streaming.
#include <functional>
#include <map>
#include <string>

namespace glm {

struct HttpRequest {
    std::string method, path, body, peer;
    std::map<std::string, std::string> headers;     // lower-case names
};

class HttpConn {
public:
    explicit HttpConn(int fd) : fd_(fd) {}
    void send(int code, const std::string& type, const std::string& body);
    void start_stream();                            // text/event-stream headers
    bool write(const std::string& data);            // false: the client is gone
    bool client_gone() const;                       // the peer closed its side (checked without blocking)
    int fd() const { return fd_; }

private:
    int fd_;
};

// the peer of this socket closed its side (checked without blocking); a descriptor already closed counts as gone
bool socket_peer_gone(int fd);

class HttpServer {
public:
    using Handler = std::function<void(const HttpRequest&, HttpConn&)>;
    // binds now (a taken port fails before the model loads); serve() runs the accept loop on a thread
    HttpServer(const std::string& host, int port);
    ~HttpServer();
    void serve(Handler h);

private:
    int lfd_ = -1;
};

}  // namespace glm
