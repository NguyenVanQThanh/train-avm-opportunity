def test_import():
    import avm_train.cli as cli

    assert cli.STAGES[0] == "fetch"
