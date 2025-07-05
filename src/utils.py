def dict_to_str(d: dict, indents=1, n_spaces=4) -> str:
    """
    Converts a dict to an indented string.
    """
    if not isinstance(d, dict):
        raise Exception(f"Argument to `dict_to_str` must be `dict`, not {type(d)}.")

    s = "{\n"

    max_key_len = float("-infinity")

    for k in d.keys():
        max_key_len = max(max_key_len, len(str(k)))

    for i, (k, v) in enumerate(d.items()):
        v_str = f'"{v}"'
        if isinstance(v, dict):
            v_str = dict_to_str(v, 1 + indents, n_spaces=n_spaces)

        s += f'{indents * " " * n_spaces}"{k}": {" " * (max_key_len - len(k))}{v_str}'

        if i != len(d) - 1:
            s += ","

        s += "\n"

    s += f"{(indents - 1) * ' ' * n_spaces}" + "}"

    return s
