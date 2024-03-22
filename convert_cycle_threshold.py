#!/usr/bin/env python
import argparse
import json
import numpy as np


LOOKUP = {
    # See 2nd paragraph, 2nd column, first page of main text (10.1001/jama.2020.3786).
    "Wang2020-thresholds": [30, 24.3],
    "Wang2020-loads": np.log10([2.6e4, 1.4e6]),
    # See supplementary table 4 (10.3346/jkms.2020.35.e86); conversion is for respiratory specimen.
    "Kim2020b-thresholds": [25.05, 35.09],
    "Kim2020b-loads": np.log10([46971053, 61341]),
}


def parse_values(values):
    """
    Parse reference or target values.
    """
    if values in LOOKUP:
        return LOOKUP[values]
    return np.asarray([float(x.strip()) for x in values.split(",")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format", choices=["plain", "json"], default="json", help="output format"
    )
    parser.add_argument(
        "--log10",
        help="whether to apply a log10 transform to the values",
        action="store_true",
    )
    parser.add_argument(
        "reference", type=parse_values, help="reference scale to convert from"
    )
    parser.add_argument(
        "target", type=parse_values, help="target scale to convert from"
    )
    parser.add_argument("values", type=parse_values, help="values to convert")
    args = parser.parse_args()

    poly = np.polynomial.Polynomial.fit(args.reference, args.target, 1)
    results = poly(np.log10(args.values) if args.log10 else args.values)

    if args.format == "json":
        results = [{"reference": x, "target": y} for x, y in zip(args.values, results)]
        print(json.dumps(results, indent=4))
    elif args.format == "plain":
        print("\n".join(map(str, results)))
    else:
        raise ValueError(args.format)
