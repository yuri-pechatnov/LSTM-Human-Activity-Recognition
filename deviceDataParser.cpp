#include <cstdio>
#include <cstdint>
#include <unistd.h>

char dd[100000];

int main() {
   // printf("s = %d\n", (int)read(0, dd, 10000000));

    struct alignas(char) TData {
        uint32_t t : 32;
        int16_t accX : 16;
        int16_t accY : 16;
        int16_t accZ : 16;
        uint8_t flags : 8;
        void out() {
            printf("%u, %d, %d, %d, %d, %d\n", (int)t, (int)accX, (int)accY, (int)accZ, !!(flags & 1), !!(flags & 2));
        }
    } data;
    const int sz = 11;
    fprintf(stderr, "sz=%u\n", sz);
    while (read(0, &data, sz) == sz) {
       data.out();
    }
    return 0;
}
