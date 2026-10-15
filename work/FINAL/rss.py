"""Run a command, then print its wall time and peak RSS (children). Usage: python3 rss.py <cmd...>"""
import resource, subprocess, sys, time
t = time.time()
rc = subprocess.run(sys.argv[1:]).returncode
print(f"[rss.py] rc={rc} wall_s={time.time() - t:.1f} peak_rss_gb={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6:.2f}")
sys.exit(rc)
