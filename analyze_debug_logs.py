#!/usr/bin/env python3
"""
Analyze debug logs from training
Usage: python analyze_debug_logs.py <debug_log_file.jsonl>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def load_log(log_file):
    """Load JSONL debug log file"""
    events = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events

def analyze_training_progression(events):
    """Analyze loss and accuracy progression"""
    training_steps = [e for e in events if e['event_type'] == 'training_step']
    
    if not training_steps:
        print("No training steps found in log")
        return
    
    print("\n=== TRAINING PROGRESSION ===")
    print(f"Total training steps logged: {len(training_steps)}")
    
    # Group by epoch
    epochs = defaultdict(list)
    for event in training_steps:
        epochs[event['epoch']].append(event)
    
    print(f"Epochs covered: {min(epochs.keys())} to {max(epochs.keys())}")
    
    # Analyze each epoch
    for epoch in sorted(epochs.keys()):
        epoch_events = epochs[epoch]
        losses = [e['data']['loss'] for e in epoch_events]
        accs = [e['data']['accuracy'] for e in epoch_events]
        lrs = [e['data']['learning_rate'] for e in epoch_events]
        
        print(f"\nEpoch {epoch}:")
        print(f"  Steps: {len(epoch_events)}")
        print(f"  Loss:  {min(losses):.4f} -> {max(losses):.4f} (avg: {statistics.mean(losses):.4f})")
        print(f"  Acc:   {min(accs):.4f} -> {max(accs):.4f} (avg: {statistics.mean(accs):.4f})")
        print(f"  LR:    {min(lrs):.6e} -> {max(lrs):.6e}")
    
    # Check for improvement
    first_10 = training_steps[:min(10, len(training_steps))]
    last_10 = training_steps[-min(10, len(training_steps)):]
    
    first_loss = statistics.mean([e['data']['loss'] for e in first_10])
    last_loss = statistics.mean([e['data']['loss'] for e in last_10])
    first_acc = statistics.mean([e['data']['accuracy'] for e in first_10])
    last_acc = statistics.mean([e['data']['accuracy'] for e in last_10])
    
    print(f"\n=== IMPROVEMENT CHECK ===")
    print(f"First 10 steps:  Loss={first_loss:.4f}, Acc={first_acc:.4f}")
    print(f"Last 10 steps:   Loss={last_loss:.4f}, Acc={last_acc:.4f}")
    print(f"Loss change:     {last_loss - first_loss:+.4f} ({'IMPROVED' if last_loss < first_loss else 'WORSE'})")
    print(f"Acc change:      {last_acc - first_acc:+.4f} ({'IMPROVED' if last_acc > first_acc else 'WORSE'})")

def analyze_gradients(events):
    """Analyze gradient statistics"""
    grad_events = [e for e in events if e['event_type'] == 'gradient_stats']
    
    if not grad_events:
        print("\nNo gradient stats found in log")
        return
    
    print("\n=== GRADIENT ANALYSIS ===")
    print(f"Gradient measurements: {len(grad_events)}")
    
    total_norms = [e['data']['total_norm'] for e in grad_events]
    max_grads = [e['data']['max_grad'] for e in grad_events]
    
    print(f"Total gradient norm:")
    print(f"  Range: {min(total_norms):.4f} to {max(total_norms):.4f}")
    print(f"  Mean:  {statistics.mean(total_norms):.4f}")
    print(f"  Median: {statistics.median(total_norms):.4f}")
    
    print(f"\nMax gradient value:")
    print(f"  Range: {min(max_grads):.6f} to {max(max_grads):.6f}")
    print(f"  Mean:  {statistics.mean(max_grads):.6f}")
    
    # Check for zero gradients
    zero_grads = [e for e in grad_events if e['data']['total_norm'] < 1e-6]
    if zero_grads:
        print(f"\n⚠️  WARNING: {len(zero_grads)} steps with near-zero gradients!")
    
    # Top modules with gradients
    all_grad_norms = defaultdict(list)
    for event in grad_events:
        for name, norm in event['data']['top_10_grad_norms'].items():
            all_grad_norms[name].append(norm)
    
    print(f"\nTop modules by average gradient norm:")
    avg_norms = [(name, statistics.mean(norms)) for name, norms in all_grad_norms.items()]
    avg_norms.sort(key=lambda x: x[1], reverse=True)
    for name, avg_norm in avg_norms[:10]:
        print(f"  {name}: {avg_norm:.4f}")

def analyze_param_updates(events):
    """Analyze parameter updates"""
    param_events = [e for e in events if e['event_type'] == 'param_updates']
    
    if not param_events:
        print("\nNo parameter update events found in log")
        return
    
    print("\n=== PARAMETER UPDATES ===")
    print(f"Update measurements: {len(param_events)}")
    
    no_updates = [e for e in param_events if e['data']['no_updates']]
    if no_updates:
        print(f"⚠️  WARNING: {len(no_updates)} measurements with NO parameter updates!")
        print(f"   Steps with no updates: {[e['step'] for e in no_updates[:10]]}")
    
    # Count params updated per step
    params_updated = [e['data']['num_params_updated'] for e in param_events]
    if params_updated:
        print(f"\nParameters updated per measurement:")
        print(f"  Range: {min(params_updated)} to {max(params_updated)}")
        print(f"  Mean:  {statistics.mean(params_updated):.1f}")
    
    # Top changing parameters
    all_changes = defaultdict(list)
    for event in param_events:
        for name, change in event['data']['top_10_param_changes'].items():
            all_changes[name].append(change)
    
    print(f"\nTop 10 parameters by average change magnitude:")
    avg_changes = [(name, statistics.mean(changes)) for name, changes in all_changes.items()]
    avg_changes.sort(key=lambda x: x[1], reverse=True)
    for name, avg_change in avg_changes[:10]:
        print(f"  {name}: {avg_change:.6e}")

def check_data_issues(events):
    """Check for data-related issues"""
    sanity_events = [e for e in events if e['event_type'] == 'data_sanity_check']
    
    if not sanity_events:
        return
    
    print("\n=== DATA SANITY CHECK ===")
    for event in sanity_events:
        data = event['data']
        print(f"Label range: [{data['label_min']}, {data['label_max']}]")
        print(f"Expected max: {data['expected_max']}")
        print(f"Labels valid: {data['labels_valid']}")
        if not data['labels_valid']:
            print("⚠️  WARNING: Labels out of valid range!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_debug_logs.py <debug_log_file.jsonl>")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"Analyzing debug log: {log_file}")
    events = load_log(log_file)
    print(f"Total events: {len(events)}")
    
    # Event type breakdown
    event_types = defaultdict(int)
    for e in events:
        event_types[e['event_type']] += 1
    
    print("\nEvent types:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    
    # Run analyses
    check_data_issues(events)
    analyze_training_progression(events)
    analyze_gradients(events)
    analyze_param_updates(events)
    
    print("\n" + "="*50)
    print("Analysis complete!")

if __name__ == '__main__':
    main()

