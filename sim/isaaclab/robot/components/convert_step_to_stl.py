"""STEP → STL 변환 스크립트"""
import sys
import cadquery as cq

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python convert_step_to_stl.py <input.step> [output.stl]")
        sys.exit(1)

    step_path = sys.argv[1]
    stl_path = sys.argv[2] if len(sys.argv) > 2 else step_path.rsplit(".", 1)[0] + ".stl"

    print(f"변환 중: {step_path} → {stl_path}")
    result = cq.importers.importStep(step_path)
    cq.exporters.export(result, stl_path)
    print("완료.")