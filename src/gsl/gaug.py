import opengsl

conf = opengsl.config.load_conf(method="grcn", dataset="cora")
dataset = opengsl.data.Dataset("cora", n_splits=1, feat_norm=conf.dataset["feat_norm"])
solver = opengsl.method.GRCNSolver(conf, dataset)
exp = opengsl.ExpManager(solver)
exp.run(n_runs=10)

