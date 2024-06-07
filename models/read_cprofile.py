import pstats

p = pstats.Stats("./profile.pstats")

p.strip_dirs().sort_stats("cumulative").print_stats()
