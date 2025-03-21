mode = "static"
configs = [
    # f"{mode}_prompt_MEDSAM",
    # f"{mode}_prompt_MEDSAMNORM",
    # f"{mode}_prompt_SAM",
    # f"{mode}_prompt_SAMNORM",
    # f"{mode}_prompt_SAM2",
    # f"{mode}_prompt_SAM2NORM",
    # f"{mode}_prompt_SAMMED3D",
    # f"{mode}_prompt_SAMMED3DNORM",
    # f"{mode}_prompt_SAMMED3DTurbo",
    # f"{mode}_prompt_SAMMED3DTurboNORM",
    # f"{mode}_prompt_SAMMED2D",
    # f"{mode}_prompt_SAMMED2DNORM",
    f"{mode}_prompt_SEGVOL",
    # f"{mode}_prompt_SEGVOLNORM",
]

# faster = {
#     "_SAM2": [4, 5],
#     "_SAM2NORM": [4, 5],
#     "_SAMMED2D": [4],
#     "_SAMMED2DNORM": [4, 5, 6],
#     "_SAMMED3DNORM": [5],
#     "_SAMMED3DTurboNORM": [5],
#     "_SAMNORM": [5],
#     "_SAM": [5],
# }
# from intrab import running_jobs


# for cnt, job in enumerate(running_jobs.file_paths):
#     if "SAMMED3D" in job:
#         VRAM = "32G"
#     else:
#         VRAM = "10G"
#     if cnt % 5 == 0:
#         print("\n")

#     sub_cmd = f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']"  -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" -gpu num=1:gmem={VRAM} -q gpu "source /home/t006d/intrab_rc && cd /dkfz/cluster/gpu/data/OE0441/t006d/Code/universal-models && python src/intrab/experiments_runner.py -c {job}" """
#     print(sub_cmd)

for config in configs:
    for ds_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:  # range(5):  # [7]:  #q
        if (mode == "interactive") and ("MEDSAM" in config):
            continue

        # if ds_id in [4, 6, 7, 10, 11]:
        #     continue
        if "SAMMED3D" in config or ds_id == 8:
            VRAM = "32G"
        else:
            VRAM = "10G"

        conf_file = config + f"_D{ds_id}"

        # for suffix in ["_faster1", "_faster2", "_faster3", "_faster4"]:
        #     conf_tmp = conf_file + suffix
        #     job_str = f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']"  -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" -gpu num=1:gmem={VRAM} -q gpu "source /home/t006d/intrab_rc && cd /dkfz/cluster/gpu/data/OE0441/t006d/Code/universal-models && python src/intrab/experiments_runner.py -c configs/{conf_tmp}.yaml" """
        #     print(job_str)
        # print("\n")
        # else:
        job_str = f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']"  -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" -gpu num=1:gmem={VRAM} -q gpu-lowprio "source /home/t006d/intrab_rc && cd /dkfz/cluster/gpu/data/OE0441/t006d/Code/universal-models && python src/intrab/experiments_runner.py -c configs/{conf_file}.yaml" """
        print(job_str)
    print("\n")
    # else:
    #     job_str = f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']"  -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" -gpu num=1:gmem={VRAM} -q gpu "source /home/t006d/intrab_rc && cd /dkfz/cluster/gpu/data/OE0441/t006d/Code/universal-models && python src/intrab/experiments_runner.py -c configs/{conf_file}.yaml" """
    #     print(job_str + "\n")
    # job_str = f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']"  -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" -gpu num=1:gmem=10.5G -q gpu "source /home/t006d/intrab_rc && cd /dkfz/cluster/gpu/data/OE0441/t006d/Code/universal-models && python src/intrab/experiments_runner.py -c configs/{conf_file}.yaml" """

    print("\n\n")
