"""Running PyTorch in a separate process (part 0.7).

The main process of the notebook never imports PyTorch. Everything related to PyTorch runs in a dedicated
process, started with the "spawn" method, which has never loaded LightGBM. The two processes only exchange
numpy arrays. This module imports neither PyTorch nor numpy when loaded: the functions run in the PyTorch
process import them in their body (PyTorch first).
"""
import atexit
import os
import signal
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

# ------------------------------------------------------------------ main process side
_ETAT = {"pool": None, "versions": None}
_MODULES = ("reseau_torch.py", "processus_torch.py")


def _versions():
    return tuple(os.path.getmtime(f) if os.path.exists(f) else None for f in _MODULES)


def _ignorer_interruptions():
    """Run when the PyTorch process starts: it ignores Ctrl-C (Jupyter's "Interrupt" button)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def arreter(immediatement=False):
    """Stops the PyTorch process (cleanly, or immediately by abandoning the running computation)."""
    pool = _ETAT["pool"]
    if pool is None:
        return
    if immediatement:
        for processus in list(getattr(pool, "_processes", {}).values()):
            processus.terminate()
    pool.shutdown(wait=not immediatement, cancel_futures=True)
    _ETAT["pool"], _ETAT["versions"] = None, None


atexit.register(arreter)


def executer(nom, *args, **kwargs):
    """Runs the function `nom` of this module in the PyTorch process and returns its result.

    The process is started at the first call (spawn method), then reused; it is restarted if one of the two
    modules has been rewritten since, so that it loads the new version. An interruption during a computation
    stops the PyTorch process, which will be restarted at the next call.
    """
    versions = _versions()
    if _ETAT["pool"] is None or _ETAT["versions"] != versions:
        arreter()
        _ETAT["pool"] = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                            initializer=_ignorer_interruptions)
        _ETAT["versions"] = versions
    try:
        return _ETAT["pool"].submit(_appeler, nom, args, kwargs).result()
    except (KeyboardInterrupt, BrokenProcessPool):
        arreter(immediatement=True)
        raise


def tester(code):
    """Runs a block of code (a test cell) in the PyTorch process and prints its output."""
    print(executer("executer_code", code), end="")


# ------------------------------------------------------------------ PyTorch process side
def _appeler(nom, args, kwargs):
    return globals()[nom](*args, **kwargs)


def executer_code(code):
    """Runs `code` in the namespace of the reseau_torch module and returns what it prints."""
    import contextlib
    import io
    import reseau_torch
    tampon = io.StringIO()
    with contextlib.redirect_stdout(tampon):
        exec(code, vars(reseau_torch))
    return tampon.getvalue()


def entrainer_ensemble(Xn, Xc, y, Xn_val, Xc_val, y_val, cardinalites, hp, lr, weight_decay, epochs,
                       batch_size, patience, n_modeles, seed):
    """Trains n_modeles ReseauPerf networks. Returns their weights (dicts of numpy arrays) and histories."""
    import torch
    from reseau_torch import ReseauPerf, entrainer_reseau

    def tenseur(a, type_):
        return None if a is None else torch.tensor(a, dtype=type_)

    Xn, Xc, y = tenseur(Xn, torch.float32), tenseur(Xc, torch.long), tenseur(y, torch.float32)
    ev = (tenseur(Xn_val, torch.float32), tenseur(Xc_val, torch.long), tenseur(y_val, torch.float32))
    poids, historiques = [], []
    for k in range(n_modeles):
        torch.manual_seed(seed + k)
        r = ReseauPerf(Xn.shape[1], cardinalites, **hp)
        h = entrainer_reseau(r, Xn, Xc, y, *ev, lr=lr, weight_decay=weight_decay, epochs=epochs,
                             batch_size=batch_size, patience=patience, seed=seed + k)
        r.eval()
        poids.append({nom: t.detach().cpu().numpy().copy() for nom, t in r.state_dict().items()})
        historiques.append(h)
    return poids, historiques


def predire_ensemble(poids, Xn, Xc, cardinalites, hp):
    """Mean prediction (standardized scale) of the networks whose weights are given."""
    import torch
    import numpy as np
    from reseau_torch import ReseauPerf
    Xn, Xc = torch.tensor(Xn, dtype=torch.float32), torch.tensor(Xc, dtype=torch.long)
    predictions = []
    with torch.no_grad():
        for etat in poids:
            r = ReseauPerf(Xn.shape[1], cardinalites, **hp)
            r.load_state_dict({nom: torch.from_numpy(t) for nom, t in etat.items()})
            r.eval()
            predictions.append(r(Xn, Xc).numpy())
    return np.mean(predictions, axis=0)
