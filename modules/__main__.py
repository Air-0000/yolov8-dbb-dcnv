"""CLI entry point for testing custom modules."""
import torch
from .dbb import DBB
from .c2f_variants import C2f_DBB, C2f_DCNV4

def main():
    print("Running self-test for custom modules...")
    x = torch.randn(2, 32, 16, 16)
    m = DBB(32, 32)
    m.eval()
    y_train = m(x)
    m.switch_to_deploy()
    y_deploy = m(x)
    diff = (y_train - y_deploy).abs().max().item()
    print(f"DBB train vs deploy max diff: {diff:.6e}")
    assert diff < 1e-4, f"DBB reparameterization diff too high: {diff}"
    print("DBB reparameterization: PASS")

    c2f = C2f_DBB(32, 32, n=1)
    assert c2f(x).shape == (2, 32, 16, 16)
    print("C2f_DBB: PASS")

    dcn4 = C2f_DCNV4(32, 32, n=1)
    assert dcn4(x).shape == (2, 32, 16, 16)
    print("C2f_DCNV4: PASS")
    print("All module verification checks passed successfully!")

if __name__ == '__main__':
    main()
