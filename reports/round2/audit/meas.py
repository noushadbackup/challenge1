import resource, subprocess, sys, time, json
t=time.time(); rc=subprocess.call(sys.argv[2:])
r=resource.getrusage(resource.RUSAGE_CHILDREN)
out={"cmd":" ".join(sys.argv[2:]),"rc":rc,"wall_s":round(time.time()-t,1),"maxrss_gb":round(r.ru_maxrss/1024**2,3),"user_s":round(r.ru_utime,1),"sys_s":round(r.ru_stime,1)}
print("MEAS", json.dumps(out)); open(sys.argv[1],"w").write(json.dumps(out))
sys.exit(rc)
