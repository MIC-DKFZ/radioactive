from tqdm import tqdm, trange
import time

# # leave all bars
# for x in trange(10, desc="loop1"):
#     for y in trange(10, desc="loop2"):
#         time.sleep(0.1)

# don't leave inner bars
for x in trange(10, desc="loop1"):
    for y in trange(10, desc="loop2", leave=False):
        time.sleep(0.1)

# # auto-detect and don't leave inner bars
# for x in trange(10, desc="loop1", leave=None):
#     for y in trange(10, desc="loop2", leave=None):
#         time.sleep(0.1)

print("\n" * 9)