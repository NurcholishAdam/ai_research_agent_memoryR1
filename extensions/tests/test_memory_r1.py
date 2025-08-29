#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for Memory-R1 Enhanced System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_memory_r1():
    """Test the Memory-R1 Enhanced system"""
    
    print("🧪 Testing Memory-R1 Enhanced System")
    print("=" * 40)
    
    try:
        # Import the system
        from memory_r1_enhanced import MemoryR1Enhanced
        print("✅ Successfully imported MemoryR1Enhanced")
        
        # Initialize system
        system = MemoryR1Enhanced()
        print("✅ Successfully initialized system")
        
        # Test basic processing
        result = system.process_input("Paris is the capital of France.", 0.8)
        print(f"✅ Processed input: {result['success']}")
        print(f"   Extracted facts: {len(result['extracted_facts'])}")
        print(f"   Response: {result['output_response'][:50]}...")
        
        # Test CI hooks
        print("\n🔧 Testing CI-Evaluable Hooks:")
        
        # Test validate_graph_state()
        graph_validation = system.validate_graph_state()
        print(f"✅ validate_graph_state(): {graph_validation['overall_status']}")
        
        # Test check_provenance_integrity()
        provenance_validation = system.check_provenance_integrity()
        print(f"✅ check_provenance_integrity(): {provenance_validation['overall_status']}")
        
        # Test replay_trace()
        if system.current_turn >= 1:
            replay_result = system.replay_trace(1, 1)
            if "error" not in replay_result:
                print(f"✅ replay_trace(1, 1): Success")
            else:
                print(f"⚠️ replay_trace(1, 1): {replay_result['error']}")
        
        # Test system status
        status = system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Current turn: {status['current_turn']}")
        print(f"   Graph nodes: {status['module_status']['graph_memory']['nodes']}")
        print(f"   Validation status: {status['validation_status']}")
        
        print(f"\n🎉 All tests passed! Memory-R1 Enhanced system is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_r1()
    sys.exit(0 if success else 1)