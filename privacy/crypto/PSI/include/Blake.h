//
// Created by fsecond on 2021/3/11.
//

#ifndef FL_PSI_BLAKE_H
#define FL_PSI_BLAKE_H

#include "Defines.h"
#include "blake2.h"
#include "memory.h"
#include "assert.h"
#include <cstring>
#include <stdexcept>

class Blake
{
public:
    // The default size of the blake digest output by Final(...);
    static const u64 HashSize = 20;

    // The maximum size of the blake digest output by Fianl(...);
    static const u64 MaxHashSize = 64;

    // Default constructor of the class. Initializes the internal state.
    Blake(u64 outputLength = HashSize) { Reset(outputLength); }

    // Resets the interal state.
    void Reset()
    {
        Reset(outputLength());
    }

    // Resets the interal state.
    void Reset(u64 outputLength)
    {

        const uint64_t blake2b_IV[8] =
                {
                        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
                        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
                };

        const unsigned char * v = (const unsigned char *)(blake2b_IV);
        std::memset(&state, 0, sizeof(blake2b_state));
        state.outlen = outputLength;
        std::memcpy(state.h, v, 64);
    }

    // Add length bytes pointed to by dataIn to the internal Blake2 state.
    template<typename T>
    typename std::enable_if<std::is_pod<T>::value>::type Update(const T* dataIn, u64 length)
    {
        unsigned char *tmp = (unsigned char *)dataIn;
        for (int i=0; i<length; i++){
            printf("%d,", *(tmp+i));
        }
        printf("\n");
        blake2b_update(&state, dataIn, length * sizeof(T));
    }

    template<typename T>
    typename std::enable_if<std::is_pod<T>::value>::type Update(const T& blk)
    {

        Update((u8*)&blk, sizeof(T));
    }

    // Finalize the Blake2 hash and output the result to DataOut.
    // Required: DataOut must be at least outputLength() bytes long.
    void Final(u8* DataOut)
    {
        assert(blake2b_final(&state, DataOut, state.outlen) == 0);
    }

    // Finalize the Blake2 hash and output the result to out.
    // Only sizeof(T) bytes of the output are written.
    template<typename T>
    typename std::enable_if<std::is_pod<T>::value && sizeof(T) <= MaxHashSize && std::is_pointer<T>::value == false>::type
    Final(T& out)
    {
        if (sizeof(T) != outputLength())
            throw std::runtime_error("Random Oracle Error");
        Final((u8*)&out);
    }

    // Copy the interal state of a Blake2 computation.
    const Blake& operator=(const Blake& src);

    // returns the number of bytes that will be written when Final(...) is called.
    u64 outputLength() const
    {
        return state.outlen;
    }
private:
    blake2b_state state;

    static blake2b_state _start_state;
};

#endif //FL_PSI_BLAKE_H
