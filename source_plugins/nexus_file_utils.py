import numpy as np

import h5py

import logging


def find_plottable_data(nexus_file):
    entries = find_default_entry(nexus_file)

    for entry in entries:
        # 1. Prioritise v3 data
        result = find_v3(entry)
        if result is not None:
            return result

        for nxdata in find_groups_recursive(entry, "NXdata"):
            # 2. Check for v2
            result = find_v2(nxdata)
            if result is not None:
                return result

            # 3. Check for v1
            signal = find_legacy_signal(nxdata)
            if signal is not None:
                axes, indices = find_v1(nxdata, signal)
                return nxdata, signal, axes, indices, "v1"

    raise ValueError("Could not find plottable NeXus data")


def read_axes(nxdata, signal, axes, indices_list=None):
    axis_values = []
    axis_labels = []

    for i, axis in enumerate(axes):
        # Determine which signal dimension(s) this axis maps to
        dim_indices = (
            indices_list[i]
            if indices_list is not None
            and i < len(indices_list)
            and indices_list[i] is not None
            else [i]
        )

        if axis == "." or not axis or axis not in nxdata:
            # Fallback to 0..N-1 index array along the mapped signal dimension
            dim_size = (
                signal.shape[dim_indices[0]] if dim_indices[0] < signal.ndim else 0
            )
            axis_values.append(np.arange(dim_size))
            axis_labels.append("")
            continue

        dataset = nxdata[axis]
        values = dataset[...]

        units = decode(dataset.attrs.get("units", ""))
        label = f"{axis} ({units})" if units else axis

        axis_values.append(values)
        axis_labels.append(label)

    return axis_values, axis_labels


def decode(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [decode(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def iterate_over_group(group):
    for name, obj in group.items():
        if isinstance(obj, h5py.Group):
            yield name, obj


def find_default_entry(root):
    if "default" in root.attrs:
        name = decode(root.attrs["default"])
        entry = root[name]

        if isinstance(entry, h5py.Group) and is_nx_class(entry, "NXentry"):
            return [entry]

    return find_groups(root, "NXentry")


def iterate_over_group_recursive(group):
    for name, obj in iterate_over_group(group):
        yield name, obj
        yield from iterate_over_group_recursive(obj)


def iterate_over_datasets_recursive(group):
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            yield name, obj

        elif isinstance(obj, h5py.Group):
            for child_name, child in iterate_over_datasets_recursive(obj):
                yield f"{name}/{child_name}", child


def decode_attrs(attr_dict: dict):
    attrs = {key: decode(value) for key, value in attr_dict.items()}
    return attrs


def find_default_group(group):
    while "default" in group.attrs:
        name = decode(group.attrs["default"])
        group = group[name]

    return group


def is_nx_class(group, class_name):
    if decode(group.attrs.get("NX_class", "")) == class_name:
        return True
    else:
        return False


def find_groups(group, nx_class_name):
    for _, child in iterate_over_group(group):
        if is_nx_class(child, nx_class_name):
            yield child


def find_groups_recursive(group, nx_class_name):
    for _, child in iterate_over_group_recursive(group):
        if is_nx_class(child, nx_class_name):
            yield child


def find_legacy_signal(nxdata):
    for name, obj in iterate_over_datasets_recursive(nxdata):
        signal = decode(obj.attrs.get("signal", 0))

        if np.any(np.asarray(signal) == 1):
            return obj

    return None


def read_plot_data(path):
    logger = logging.getLogger("main")
    with h5py.File(path, "r") as nexus_file:
        nxdata, signal, axes, indices, version = find_plottable_data(nexus_file)

        data = np.empty(signal.shape, dtype=signal.dtype)
        logger.info(f"You are using {version} data.")
        if signal.chunks:
            for slice in signal.iter_chunks():
                signal.read_direct(data, source_sel=slice, dest_sel=slice)
        else:
            signal.read_direct(
                data
            )  # maybe not load in everything at once? unsure of alternatives

        axis_values, axis_labels = read_axes(nxdata, signal, axes)

        return data, axis_labels, indices, axis_values


def _extract_v3_nxdata(nxdata: h5py.Group):
    if not isinstance(nxdata, h5py.Group) or not is_nx_class(nxdata, "NXdata"):
        return None

    signal_name = decode(nxdata.attrs.get("signal", ""))
    if not signal_name or signal_name not in nxdata:
        return None

    signal = nxdata[signal_name]
    axes = parse_axes(nxdata.attrs.get("axes", []))

    # If @axes is omitted entirely, default to "." for each signal dimension
    if not axes:
        axes = ["."] * signal.ndim

    indices_list = []
    for i, axis in enumerate(axes):
        if axis == "." or not axis:
            indices_list.append([i] if i < signal.ndim else None)
            continue

        raw_indices = nxdata.attrs.get(f"{axis}_indices", None)
        if raw_indices is not None:
            indices = [int(idx) for idx in np.atleast_1d(raw_indices)]
        else:
            indices = [i] if i < signal.ndim else None

        indices_list.append(indices)

    return nxdata, signal, axes, indices_list, "v3"


def find_v3(entry: h5py.Group):
    logger = logging.getLogger("main")

    if "default" in entry.attrs:
        name = decode(entry.attrs["default"])
        if name in entry:
            result = _extract_v3_nxdata(entry[name])
            if result is not None:
                return result
        logger.warning(f"Default NXdata target '{name}' is invalid or missing signal.")

    for nxdata in find_groups(entry, "NXdata"):
        result = _extract_v3_nxdata(nxdata)
        if result is not None:
            return result

    return None


def parse_axes(value):
    value = decode(value)

    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]

    if isinstance(value, np.ndarray):
        return [decode(v) for v in value.tolist()]

    if isinstance(value, list):
        return [decode(v) for v in value]

    return []


def find_v2(nxdata: h5py.Group):
    for name, obj in nxdata.items():
        if not isinstance(obj, h5py.Dataset):
            continue

        raw_signal = decode(obj.attrs.get("signal", 0))
        # Converts scalar, list, or ndarray to an int array and checks if any element == 1
        try:
            is_signal = np.any(np.asarray(raw_signal, dtype=int) == 1)
        except (ValueError, TypeError):
            is_signal = False

        if is_signal:
            axes = parse_axes(obj.attrs.get("axes", []))
            indices = [[i] for i in range(len(axes))]
            return nxdata, obj, axes, indices, "v2"
    return None


def find_v1(nxdata: h5py.Group, signal):
    rank = signal.ndim
    axes = ["."] * rank
    indices = [[i] for i in range(rank)]

    for name, obj in iterate_over_datasets_recursive(nxdata):
        axis = decode(obj.attrs.get("axis", 0))
        primary = decode(obj.attrs.get("primary", 0))

        if not axis or not np.any(np.asarray(primary) == 1):
            continue

        axis = int(axis)
        if 1 <= axis <= rank:
            axes[axis - 1] = name

    return axes, indices
