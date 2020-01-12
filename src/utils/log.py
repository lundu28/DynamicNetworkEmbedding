import logging

def get_logger(log_filename=None, module_name=__name__, level=logging.INFO):

    # select handler
    if log_filename is None:
        handler = logging.StreamHandler()
    elif type(log_filename) is str:
        handler = logging.FileHandler(log_filename, 'w')
    else:
        raise ValueError("log_filename invalid!")

    # build logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler.setLevel(level)
    formatter = logging.Formatter(('%(asctime)s %(filename)s' \
                    '[line:%(lineno)d] %(levelname)s %(message)s'))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def serialize_tree_level(tree):
    level_dic = {}
    def dfs(u, dep = 0):
        if dep not in level_dic:
            level_dic[dep] = []
        s = "id: %s, child: " % tree[u].id
        for i in tree[u].childst:
            s += str(i) + ", "
        s = s[: -2]
        s += "\n"
        level_dic[dep].append(s)
        for i in tree[u].childst:
            dfs(i, dep + 1)
    dfs(len(tree) - 1)
    s = ""
    for i in level_dic:
        s += "level %d: \n" % i
        for j in level_dic[i]:
            s += j
        s += "\n"
    return s
