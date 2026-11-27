import os
import numpy as np

class SpikeRouterGoldenModel:
    """
    Bit-Exact Golden Reference Model for Edge AI SoC Target Router:
      - Maps N source lanes to M target destinations via a programmable routing table
      - Supports Unicast (1-to-1), Multicast (1-to-Many), and Dropped/Masked routes
      - Outputs deterministic target vectors per cycle
    """
    def __init__(self, num_sources: int, num_targets: int, routing_table: dict):
        self.num_sources = int(num_sources)
        self.num_targets = int(num_targets)
        # routing_table maps source_id (int) -> list of target_ids (list[int])
        self.routing_table = {
            int(src): [int(t) for t in targets if 0 <= int(t) < self.num_targets]
            for src, targets in routing_table.items()
            if 0 <= int(src) < self.num_sources
        }

    def route_step(self, active_spikes: list, mode: str = "binary"):
        """
        Routes a vector of source values (length = num_sources) to target ports (length = num_targets).
          - mode="binary": OR-reduction of binary spikes arriving at each target (0 or 1)
          - mode="accumulate": Sums signed INT8 payloads arriving at each target (clamped to INT8)
        """
        assert len(active_spikes) == self.num_sources, f"Expected {self.num_sources} source inputs."

        target_bus = np.zeros(self.num_targets, dtype=np.int32)

        for src_id, val in enumerate(active_spikes):
            if val != 0 and src_id in self.routing_table:
                for dst_id in self.routing_table[src_id]:
                    if mode == "binary":
                        target_bus[dst_id] = 1
                    else:
                        target_bus[dst_id] += int(val)

        if mode == "binary":
            return target_bus.astype(np.uint8).tolist()
        else:
            return np.clip(target_bus, -128, 127).astype(np.int8).tolist()


def run_router_verification():
    # Architecture: 4 Source Neurons/Lanes -> 4 Target Destinations
    # Target 0: Hidden Neuron A | Target 1: Hidden Neuron B
    # Target 2: Output Classifier | Target 3: Wake-Up Logic Unit
    routing_map = {
        0: [0, 3],     # Source 0 -> Multicast to Target 0 AND Target 3 (Wake-Up)
        1: [1],        # Source 1 -> Unicast to Target 1
        2: [0, 1, 2],  # Source 2 -> Broadcast to Targets 0, 1, and 2
        3: [],         # Source 3 -> Masked / Pruned (Dropped)
    }

    router = SpikeRouterGoldenModel(num_sources=4, num_targets=4, routing_table=routing_map)

    # Test Vectors: (Description, Source Spike Vector [S0, S1, S2, S3], Expected Target Vector [T0, T1, T2, T3])
    test_cases = [
        ("Idle Cycle (All Zeros)",           [0, 0, 0, 0], [0, 0, 0, 0]),
        ("Unicast Route (S1 -> T1)",         [0, 1, 0, 0], [0, 1, 0, 0]),
        ("Multicast + Wake-Up (S0 -> T0,T3)",[1, 0, 0, 0], [1, 0, 0, 1]),
        ("Masked/Pruned Source (S3 dropped)",[0, 0, 0, 1], [0, 0, 0, 0]),
        ("3-Way Fan-Out (S2 -> T0,T1,T2)",   [0, 0, 1, 0], [1, 1, 1, 0]),
        ("Simultaneous Collision (S0 + S1)", [1, 1, 0, 0], [1, 1, 0, 1]),
        ("All Sources Active (Full Burst)",  [1, 1, 1, 1], [1, 1, 1, 1]),
    ]

    output_path = os.path.expanduser("~/edge_ai_soc_lab/router_verification_results.txt")

    with open(output_path, "w") as f:
        header = (
            "EDGE AI SOC TARGET ROUTER GOLDEN REFERENCE SIMULATION\n"
            + "=" * 98 + "\n"
            f"Config: 4 Sources -> 4 Targets | Routing Map: {routing_map}\n"
            + "-" * 98 + "\n"
            f"{'Step':<5} | {'Test Case':<34} | {'Source In':<14} | {'Expected':<14} | {'Actual':<14} | {'Status'}\n"
            + "-" * 98 + "\n"
        )
        print(header, end="")
        f.write(header)

        passed = 0
        for step, (desc, src_vec, exp_vec) in enumerate(test_cases):
            act_vec = router.route_step(src_vec, mode="binary")
            is_pass = (act_vec == exp_vec)
            status = "PASS" if is_pass else "FAIL"
            if is_pass:
                passed += 1

            row = f"{step:<5} | {desc:<34} | {str(src_vec):<14} | {str(exp_vec):<14} | {str(act_vec):<14} | {status}\n"
            print(row, end="")
            f.write(row)

        footer = (
            "-" * 98 + "\n"
            f"Summary: {passed}/{len(test_cases)} Router Verification Tests Passed.\n"
        )
        print(footer, end="")
        f.write(footer)

    print(f"\nSaved standardized router verification file to '{output_path}'.")


if __name__ == "__main__":
    run_router_verification()
