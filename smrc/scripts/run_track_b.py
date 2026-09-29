"""Track B: simulated quantisation of all three classifiers."""
import _common  # noqa: F401  (must come first)

from src import config, data
from src.baselines import fit_tuned
from src.metrics import evaluate
from src.quantisation import count_parameters, packed_size_kb, quantise_model
from src.utils import log_result, print_row, set_seed


def main():
    args = _common.parse_args(__doc__)
    set_seed(config.SEED)
    ds = data.load(args.data)
    X_train, X_test, _ = ds.vectorise()

    for key in config.QUANT_MODELS:
        base, _ = fit_tuned(key, X_train, ds.y_train, config.SEED)
        n_params = count_parameters(base)
        # 32 bits is the unquantised reference.
        for bits in [32, *config.QUANT_BITS]:
            model = base if bits == 32 else quantise_model(base, bits, config.QUANT_SCHEME)
            row = {"seed": config.SEED, "model": key, "track": "B", "variant": "quantise",
                   "bits": bits, "n_params": n_params,
                   "packed_size_kb": packed_size_kb(n_params, bits),
                   **evaluate(model, X_test, ds.y_test)}
            log_result(row)
            print_row(row)

    print(f"\nLogged to {config.RESULTS_CSV}")


if __name__ == "__main__":
    main()
