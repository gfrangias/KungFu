#include <kungfu/peer.hpp>
#include <libkungfu-comm.h>
#include <stdio.h>
#include <cstdlib>

namespace kungfu
{
std::unique_ptr<Peer> &Peer::GetDefault(bool reinit)
{
    static std::unique_ptr<Peer> instance;
    if (reinit) { instance.reset(new Peer); }
    return instance;
}

Peer::Peer()
{
    const int err = GoKungfuInit();
    if (err) {
        fprintf(stderr, "%s failed\n", "GoKungfuInit");
        exit(1);
    }
}

Peer::Peer(int rank, int size)
{
    const int err = GoKungfuInitSingleMachine(GoInt(rank), GoInt(size));
    if (err) {
        fprintf(stderr, "%s failed\n", "GoKungfuInitSingleMachine");
        exit(1);
    }
}

Peer::~Peer() { GoKungfuFinalize(); }

bool Peer::Detached() const { return GoKungfuDetached(); }

uint64_t Peer::Uid() const { return GoKungfuUID(); }

int Peer::Noop(const DoneCallback &done)
{
    return GoKungfuNoop(new CallbackWrapper(done));
}

int Peer::Save(const char *name, const void *buf, int count,
               KungFu_Datatype dtype)
{
    return GoKungfuSave(const_cast<char *>(name), const_cast<void *>(buf),
                        GoInt(count), dtype, nullptr);
}

int Peer::Save(const char *name, const void *buf, int count,
               KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuSave(const_cast<char *>(name), const_cast<void *>(buf),
                        GoInt(count), dtype, new CallbackWrapper(done));
}

int Peer::Save(const char *version, const char *name, const void *buf,
               int count, KungFu_Datatype dtype)
{
    return GoKungfuSaveVersion(
        const_cast<char *>(version), const_cast<char *>(name),
        const_cast<void *>(buf), GoInt(count), dtype, nullptr);
}

int Peer::Save(const char *version, const char *name, const void *buf,
               int count, KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuSaveVersion(const_cast<char *>(version),
                               const_cast<char *>(name),
                               const_cast<void *>(buf), GoInt(count), dtype,
                               new CallbackWrapper(done));
}

int Peer::Request(int destRank, const char *name, void *buf, int count,
                  KungFu_Datatype dtype)
{
    return GoKungfuRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), dtype, nullptr);
}

int Peer::Request(int destRank, const char *name, void *buf, int count,
                  KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), dtype, new CallbackWrapper(done));
}

int Peer::Request(int rank, const char *version, const char *name, void *buf,
                  int count, KungFu_Datatype dtype)
{
    return GoKungfuRequestVersion(rank, const_cast<char *>(version),
                                  const_cast<char *>(name), buf, GoInt(count),
                                  dtype, nullptr);
}

int Peer::Request(int rank, const char *version, const char *name, void *buf,
                  int count, KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuRequestVersion(rank, const_cast<char *>(version),
                                  const_cast<char *>(name), buf, GoInt(count),
                                  dtype, new CallbackWrapper(done));
}

// monitoring APIs
int Peer::GetPeerLatencies(float *recvbuf, int recv_count)
{
    return GoKungfuGetPeerLatencies(recvbuf, recv_count, KungFu_FLOAT);
}

int Peer::CheckInterference() { return GoKungfuCheckInterference(); }

// control APIs
int Peer::ResizeCluster(const uint32_t new_size, bool *changed, bool *detached)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    return GoKungfuResizeCluster(GoInt(new_size),
                                 reinterpret_cast<char *>(changed),
                                 reinterpret_cast<char *>(detached));
}

int Peer::ResizeClusterFromURL(bool *changed, bool *detached)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    return GoKungfuResizeClusterFromURL(reinterpret_cast<char *>(changed),
                                        reinterpret_cast<char *>(detached));
}

int Peer::ChangeCluster(const uint64_t progress, bool *changed, bool *detached)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    return GoKungfuChangeCluster(GoInt(progress),
                                 reinterpret_cast<char *>(changed),
                                 reinterpret_cast<char *>(detached));
}

int Peer::InitProgress() const
{
    int progress;
    GoKungfuInitProgress(&progress);
    return progress;
}

void Peer::LogStats() { GoLogStats(); }

void Peer::CalcStats() { GoCalcStats(); }

void Peer::PrintStategyStats() { GoPrintStategyStats(); }

int Peer::ProposeNewSize(int new_size)
{
    return GoKungfuProposeNewSize(GoInt(new_size));
}
}  // namespace kungfu
