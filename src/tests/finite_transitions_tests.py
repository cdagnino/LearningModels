import numpy as np
import src


def test_update_lambdas():
    new_lambdas = src.update_lambdas(2.1, src.dmd_transition_fs, np.array([0., 0., 1.]),
                                     action=0.8, old_state="meh")
    assert np.isfinite(new_lambdas).all()


def test_belief():
    belief_at_x = src.belief(3.1, src.dmd_transition_fs,
                         [0., 0., 1.], 3.0, old_state='meh')
    assert isinstance(belief_at_x, float)
    assert np.isfinite(belief_at_x)

    belief_at_x = src.belief(2.1, src.dmd_transition_fs,
                         [0.3, 0.4, 0.3], 2.8, old_state='meh')
    assert isinstance(belief_at_x, float)
    assert np.isfinite(belief_at_x)

    print("new belief f tested")

