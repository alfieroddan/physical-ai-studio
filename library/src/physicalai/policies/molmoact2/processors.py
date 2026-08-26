class MolmoAct2PreProcessor:
    def __init__(self) -> None:
        pass

class MolmoAct2PostProcessor:
    def __init__(self) -> None:
        pass


def make_policy_processors(config):
    return MolmoAct2PreProcessor(), MolmoAct2PostProcessor()