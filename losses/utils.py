
def l1l2(x):
    return x.abs() + (x ** 2.0)

def get_color_from_state(A):
    return A[...,-3:] + 0.5

def snake_to_pascal(snake_case_str: str):
    temp_str = snake_case_str.replace("_", " ")
    title_case_str = temp_str.title()
    pascal_case_str = title_case_str.replace(" ", "")
    return pascal_case_str

def parse_slice_string(slice_str: str) -> slice:
    parts = slice_str.split(':')
    def int_or_none(s):
        s = s.strip()
        return int(s) if s != '' else None
    args = [int_or_none(part) for part in parts]
    return slice(*args)
