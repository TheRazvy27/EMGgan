from generate_signals import SimplifiedEMGGenerator
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    print("="*60)
    print("GENERATING FINAL EMG SIGNALS")
    print("="*60)
    
    # Initialize generator
    generator = SimplifiedEMGGenerator(Path('.'))
    
    # Create output directory
    output_dir = Path('final_simplified_signals')
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i in range(100):  # Change this number to generate more signals
        print(f'Generating signal {i+1}/100...')
        
        # Generate signal
        signal_data = generator.generate_realistic_signal(300)
        validation = generator.validate_signal(signal_data)
        
        # Save signal
        filename = f'final_simplified_{i+1:02d}_onset_{signal_data["onset_time"]:.0f}s.npy'
        filepath = output_dir / filename
        np.save(filepath, signal_data['signal'])
        
        # Check direction match
        target_negative = signal_data['target_mnf_slope'] < 0
        actual_negative = validation['mnf_trend']['slope'] < 0
        direction_match = target_negative == actual_negative
        
        # Print result
        status = "SUCCESS" if direction_match else "PARTIAL"
        print(f'  {status}: Target: {signal_data["target_mnf_slope"]:.4f}, Actual: {validation["mnf_trend"]["slope"]:.4f}')
        
        # Store results
        results.append({
            'Filename': filename,
            'Expected_Onset': signal_data['onset_time'],
            'Detected_Onset': validation['fatigue_result']['onset_time'],
            'Target_MNF_Slope': signal_data['target_mnf_slope'],
            'Actual_MNF_Slope': validation['mnf_trend']['slope'],
            'MNF_Trend': validation['mnf_trend']['trend'],
            'Detection_Confidence': validation['fatigue_result']['confidence'],
            'Direction_Match': direction_match
        })
    
    # Save results
    df = pd.DataFrame(results)
    results_path = 'final_simplified_validation.csv'
    df.to_csv(results_path, index=False, float_format='%.4f')
    
    # Calculate final statistics
    direction_success = df['Direction_Match'].sum()
    successful_detections = len(df[pd.to_numeric(df['Detected_Onset'], errors='coerce').notna()])
    
    if successful_detections > 0:
        detection_errors = pd.to_numeric(df['Detected_Onset'], errors='coerce') - df['Expected_Onset']
        avg_error = detection_errors.abs().mean()
    else:
        avg_error = float('inf')

    print(f"\n" + "="*60)
    print("FINAL GENERATION COMPLETE")
    print("="*60)
    print(f"Signals saved to: {output_dir}")
    print(f"Results saved to: {results_path}")
    print(f"\nFINAL STATISTICS:")
    print(f"  Total signals: 100")
    print(f"  Direction matches: {direction_success}/100 ({direction_success/100*100:.1f}%)")
    print(f"  Successful detections: {successful_detections}/100 ({successful_detections/100*100:.1f}%)")
    print(f"  Average detection error: {avg_error:.1f} seconds")
    print(f"  MNF slope range: {df['Actual_MNF_Slope'].min():.4f} to {df['Actual_MNF_Slope'].max():.4f}")

if __name__ == "__main__":
    main() 
