SOURCE=/mnt/c/Users/Henry/Documents/unity-stuff/labUX/bancada_invertida.pcd
TARGET=/mnt/c/Users/Henry/Documents/unity-stuff/target-i-04.pcd
run:
	pipenv run python -m apps.align_point_cloud $(SOURCE) $(TARGET) --logdir logs/LCD-D256

run-trans:
	pipenv run python -m apps.translate_pcd $(SOURCE) $(TARGET)

serve-dev:
	cd lcd && pipenv run uvicorn main:app --reload --host 0.0.0.0 --port 8080

serve:
	cd lcd && pipenv run uvicorn main:app --host 0.0.0.0 --port 8080
