"""WWW handling things."""

from datetime import datetime, time
from pathlib import Path

import os
import requests
import yaml

## build_meter_data
# @since		2026-03-23 21:30:12
def build_meter_data( **kwargs ):
	latest_dir = kwargs.get( 'latest_dir' )
	meter_filenames = {}
	for camera_id in kwargs.get( 'meter_configs' ):
		filename = latest_dir / ( camera_id + ".jpg" )
		meter_filenames[ camera_id ] = filename
	kwargs[ 'meter_filenames' ] = meter_filenames
	return kwargs

## Not my code. Thanks grok!
# @since		2026-03-23 20:57:18
def is_time_between(start_time_str: str, end_time_str: str) -> bool:
	"""
	Check if current time is between two time strings in 24-hour format.
	Example: "19:34" and "23:55"

	Returns True if current time is ≥ start and ≤ end (same day)
	"""
	try:
		# Parse the time strings
		start = datetime.strptime(start_time_str, "%H:%M").time()
		end	  = datetime.strptime(end_time_str,	  "%H:%M").time()

		# Get current time (only hour and minute)
		now = datetime.now().time()

		# Case 1: Normal range (start < end) → e.g. 09:00–17:00
		if start <= end:
			return start <= now <= end

		# Case 2: Overnight range (start > end) → e.g. 22:00–06:00
		else:
			return now >= start or now <= end

	except ValueError as e:
		print(f"Error parsing time: {e}")
		return False

## Search for the latest images.
# @since		2026-03-23 20:57:58
def latest_images_exist( **kwargs ):
	# Do all of the images exist?
	meter_filenames = kwargs.get( 'meter_filenames' )
	for camera_id, filename in meter_filenames.items():
		if not filename.exists():
			print( filename, "does not exist" )
			return False
	return True

## Load the www config from disk.
# @since		2026-03-23 20:56:48
def load_config():
	www_path = Path( "configs/www.yaml" )
	with www_path.open() as f:
		data = yaml.safe_load(f)
		return data[ 'www' ]

## Handle the uploading of the latest images.
# @since		2026-03-23 20:57:01
def maybe_upload_latest_images( **kwargs ):
	kwargs[ 'www_config' ] = load_config()
	kwargs = build_meter_data( **kwargs )
	if not should_we_upload( **kwargs ):
		return False
	upload_the_images( **kwargs )

## Decide whether we should upload the latest images.
# @since		2026-03-23 20:56:33
def should_we_upload( **kwargs ):

	www_config = kwargs.get( 'www_config' )

	# Uploading must be enabled at all
	if not www_config[ 'image_upload_enabled' ]:
		return False

	# And the images must exist.
	if not latest_images_exist( **kwargs ):
		return False

	# And now we check the time.
	if not is_time_between( www_config[ 'image_upload_begin' ], www_config[ 'image_upload_end' ] ):
		return False

	# It all looks good.
	return True

def upload_the_images( **kwargs ):

	files = {}
	values = {}

	meter_filenames = kwargs.get( 'meter_filenames' )
	www_config = kwargs.get( 'www_config' )

	values[ "water_values" ] = "receive_latest_images"

	for camera_id, filename in meter_filenames.items():
		files[ camera_id ] = open( filename, 'rb' )
		values[ "timestamp" ] = int( os.path.getmtime( filename ) )

	r = requests.post(www_config[ 'url' ], files=files, data=values )
	print( r.text )
