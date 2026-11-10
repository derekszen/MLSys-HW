"""Parse the observed npu-smi process table; unknown output is not a free chip."""


def occupied_chips(text):
    marker = "Process memory(MB)"
    if text.count(marker) != 1:
        raise ValueError("missing or ambiguous NPU process-table header")
    # The marker is inside the header line; discard the rest of that line.
    lines = text.split(marker, 1)[1].splitlines()[1:]
    occupied = set()
    explicit_empty = False
    count = 0
    for line in lines:
        line = line.strip()
        if not line or set(line) <= set("+-="):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if cells == ["No running processes found"]:
                explicit_empty = True
                continue
            if len(cells) == 4:
                location = cells[0].split()
                if (len(location) == 2 and all(x.isascii() and x.isdecimal() for x in location)
                        and cells[1].isascii() and cells[1].isdecimal() and int(cells[1]) > 0
                        and cells[2] and cells[3].isascii() and cells[3].isdecimal()):
                    occupied.add(tuple(map(int, location)))
                    count += 1
                    continue
        raise ValueError("unrecognized row in NPU process table")
    if explicit_empty and count:
        raise ValueError("conflicting empty and occupied process table")
    if not count and not explicit_empty:
        raise ValueError("process table has no rows or explicit empty marker")
    return occupied
