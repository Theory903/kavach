"""Bootstrap training for the Reinforcement Learning Advisor.

Runs the bundled Kavach training dataset through the Gateway and simulates
human-level feedback to the RL engine to pre-populate the Q-table.
"""

from __future__ import annotations

import logging
import random
import time

from kavach.core.gateway import KavachGateway
from kavach.ml.dataset import TRAINING_DATA, BENIGN

logger = logging.getLogger(__name__)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Enable metrics/policies
    gateway = KavachGateway()
    advisor = gateway._rl_advisor
    
    print("="*60)
    print("RL BOOTSTRAP TRAINER")
    print(f"Target: {advisor._persist_path}")
    print(f"Initial updates: {advisor.total_updates}")
    print("="*60)
    
    total_samples = len(TRAINING_DATA)
    epochs = 100
    total_steps = 0
    start = time.monotonic()
    
    print(f"Training on {total_samples} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Shuffle for stochasticity
        data = list(TRAINING_DATA)
        random.shuffle(data)
        
        for sample in data:
            is_attack = sample.label != BENIGN
            
            # 1. Forward pass (gateway evaluation)
            result = gateway.analyze(
                prompt=sample.text,
                user_id="rl_trainer_bot",
                role=random.choice(["guest", "default", "admin"])  # simulate roles
            )
            
            action_taken = result.get("decision", "allow")
            risk_score = result.get("risk_score", 0.0)
            
            # The exact RL state hit during inference
            rl_sugg = result.get("rl_suggestion", {})
            state_index = rl_sugg.get("state_index")
            action_index = rl_sugg.get("action_index")
            
            if state_index is not None and action_index is not None:
                # 2. Compute true reward based on the label
                was_blocked = action_taken in ("block", "sanitize")
                reward = advisor.compute_reward(
                    action_taken=action_taken,
                    actual_is_attack=is_attack,
                    was_blocked=was_blocked,
                )
                
                # 3. Backward pass (Q-table update)
                advisor.update(
                    state_index=state_index,
                    action_index=action_index,
                    reward=reward
                )
                total_steps += 1
                
    advisor.save()
    
    elapsed = time.monotonic() - start
    stats = advisor.get_stats()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Elapsed Time:     {elapsed:.2f}s")
    print(f"  Total Updates:    {stats['total_updates']} ({total_steps} new)")
    print(f"  Non-Zero Cells:   {stats['non_zero_cells']} / {stats['q_table_cells']}")
    print(f"  State Coverage:   {stats['coverage_pct']}%")
    print(f"  Model Saved To:   {advisor._persist_path}")
    print()

if __name__ == "__main__":
    main()
