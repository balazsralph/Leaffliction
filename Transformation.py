def main() -> None:
	if len(sys.argv) != 2:
		print("Wrong number of arguments")
		sys.exit(1)
	
	picture_path = Path(sys.argv[1])
	if picture_path.suffix.lower() not in PICTURE_EXTENSIONS or not picture_path.is_file():
		print("Not a picture")
		sys.exit(1)
	
	image = cv2.imread(image_path)

if __name__ == "__main__":
    main()