from qepc.notebook_header import qepc_notebook_setup

env = qepc_notebook_setup(run_diagnostics=False)  # set True in 00_setup if you want
data_dir = env.data_dir
raw_dir = env.raw_dir
