#include <iostream>

int main() {
    int values[] = {5, 3, 8, 1, 4};
    int count = sizeof(values) / sizeof(values[0]);

    for (int i = 0; i < count - 1; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (values[j] < values[i]) {
                int temp = values[i];
                values[i] = values[j];
                values[j] = temp;
            }
        }
    }

    std::printf("Sorted values:");
    for (int i = 0; i < count; ++i) {
        std::printf(" %d", values[i]);
    }
    std::printf("\n");

    return 0;
}

