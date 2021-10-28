SOURCE=/mnt/c/Users/Henry/Documents/unity-stuff/QuartoHenryFullWithTotem_md_color.pcd
TARGET=/mnt/c/Users/Henry/Documents/unity-stuff/mobile_pcd_18_espelhado.pcd
run:
	pipenv run python -m apps.align_point_cloud $(SOURCE) $(TARGET) --logdir logs/LCD-D256
serve:
	cd lcd && pipenv run uvicorn main:app --reload