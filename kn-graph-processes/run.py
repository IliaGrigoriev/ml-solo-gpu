from build_graph            import BuildGraph
from message_passing_kernel import MsgPassingKernel
from process_level_emb      import ProcessLevelEmb

from paths import SAP_SAM

if __name__ == "__main__":
    build_graph = BuildGraph()
    pr_lev_emb  = ProcessLevelEmb()

    graph_builder = BuildGraph()
    mess_pass     = MsgPassingKernel()

    pr_lev_emb.embed(SAP_SAM, graph_builder, mess_pass)