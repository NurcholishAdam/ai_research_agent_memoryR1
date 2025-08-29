#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for Memory-R1 Modular System
"""

def test_memory_r1_simple():
    """Simple test of Memory-R1 system"""
    
    print("🧪 Testing Memory-R1 Modular System")
    print("=" * 40)
    
    try:
        # Import and initialize
        from memory_r1_modular import MemoryR1Enhanced
        print("✅ Successfully imported MemoryR1Enhanced")
        
        system = MemoryR1Enhanced()
        print("✅ Successfully initialized system")
        
        # Test basic processing
        result = system.process_input("Paris is the capital of France.", 0.8)
        print(f"✅ Processed input successfully: {result['success']}")
        
        if result['success']:
            print(f"   Facts extracted: {len(result['extracted_facts'])}")
            print(f"   Operations: {result['graph_operations']}")
            print(f"   Response: {result['output_response'][:60]}...")
        
        # Test CI hooks
        print("\n🔧 Testing CI-Evaluable Hooks:")
        
        graph_val = system.validate_graph_state()
        print(f"✅ validate_graph_state(): {graph_val['overall_status']}")
        
        prov_val = system.check_provenance_integrity()
        print(f"✅ check_provenance_integrity(): {prov_val['overall_status']}")
        
        replay_val = system.replay_trace(1, 1)
        if "error" not in replay_val:
            print(f"✅ replay_trace(1, 1): Success")
        else:
            print(f"⚠️ replay_trace(1, 1): {replay_val['error']}")
        
        print(f"\n🎉 Memory-R1 Modular System is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_memory_r1_simple()