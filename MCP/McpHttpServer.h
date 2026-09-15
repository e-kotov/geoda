/**
 * GeoDa TM, Copyright (C) 2011-2025 by Luc Anselin - all rights reserved
 *
 * This file is part of GeoDa.
 *
 * GeoDa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GeoDa is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __GEODA_CENTER_MCP_HTTP_SERVER_H__
#define __GEODA_CENTER_MCP_HTTP_SERVER_H__

#include <atomic>
#include <map>
#include <set>
#include <string>
#include <wx/event.h>
#include <wx/socket.h>
#include <wx/thread.h>

class McpServer;

// Minimal HTTP/1.1 server on wxSocketServer bound to 127.0.0.1. Serves the
// MCP protocol (JSON-RPC 2.0) on POST /mcp, a no-op response on OPTIONS /mcp,
// and a health check on GET /. No CORS headers are sent: the server is meant
// for desktop MCP clients, and browsers must not be able to reach it.
//
// Threading: light tools (project/status, table/*, weights/*) and window
// tools (window/create_map, window/create_plot -- wx window creation is
// main-thread-only) are handled synchronously on the main thread. Heavy tools
// (LISA with permutations, clustering) spawn a worker wxThread that computes
// the result and writes the response to the handed-off socket; the main thread
// never touches the socket after handoff. At most kMaxWorkers heavy tools run
// concurrently; further heavy requests are rejected with 503.
//
// Lifetime: every socket this server owns is tracked in m_clients, so Stop()
// can disarm and destroy all of them. A wxSocketBase left open keeps its
// CFSocket registered with the run loop and calls back into its event handler
// (this object); if the handler is destroyed first, the next event that pumps
// -- e.g. the modal dialog a buffered wxLogGui raises during shutdown --
// dereferences a dead wxEvtHandler and the process crashes. Stop() therefore
// also waits for in-flight workers before releasing the socket and the
// McpServer they borrowed.
class McpHttpServer : public wxEvtHandler
{
public:
    // port 0 = let the OS auto-assign a free port.
    McpHttpServer(int port = 0);
    ~McpHttpServer();

    bool Start();
    void Stop();
    bool IsRunning() const { return m_server != NULL; }
    int GetPort() const { return m_port; }
    wxString GetUrl() const;

private:
    // Maximum number of concurrent heavy-tool worker threads.
    static const int kMaxWorkers = 4;

    // State of one client connection. Tracked from the moment the socket is
    // accepted, not from the first byte read: a connection that never sends a
    // complete request must still be destroyed by Stop().
    struct Client {
        Client() : handed_off(false) {}
        std::string buffer;  // request bytes received so far
        bool handed_off;     // a worker owns the socket until it posts back
    };
    typedef std::map<wxSocketBase*, Client> ClientMap;

    void OnServerEvent(wxSocketEvent& event);
    void OnClientEvent(wxSocketEvent& event);
    // Closes and destroys a socket handed back from a worker thread. wxSocket
    // on macOS must be closed on the thread that created it (the main thread).
    void OnSocketClose(wxCommandEvent& event);
    void HandleRequest(wxSocketBase* socket, const std::string& method,
                       const std::string& path, const std::string& body);
    void SendResponse(wxSocketBase* socket, const std::string& body,
                      int status);
    void SendCorsPreflight(wxSocketBase* socket);
    void SendHealth(wxSocketBase* socket);
    // Disarm, close and destroy a tracked socket, dropping it from m_clients.
    // The only place a client socket is destroyed; safe to call twice.
    void Retire(wxSocketBase* socket);
    // Wait for and delete workers whose Entry() has already returned.
    void ReapWorkers();

    wxSocketServer* m_server;
    int m_port;
    McpServer* m_mcp;
    // Every client socket we own, with its accumulated request bytes.
    ClientMap m_clients;
    // Joinable heavy-tool workers, waited on by Stop() and reaped as they
    // finish. Detached threads could not be waited for, which left a worker
    // holding a socket and m_mcp past their destruction.
    std::set<wxThread*> m_workers;
    // Number of heavy-tool workers currently running (bounded by kMaxWorkers).
    std::atomic<int> m_active_workers;
    // Set by Stop(); read by workers so they do not post to a handler that is
    // going away. Cleared by Start().
    std::atomic<bool> m_stopping;

    wxDECLARE_NO_COPY_CLASS(McpHttpServer);
    wxDECLARE_EVENT_TABLE();
};

#endif
