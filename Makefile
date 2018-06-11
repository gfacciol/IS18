default: srtm

srtm: 
	$(MAKE) -C srtm4
	mv srtm4/srtm4 /usr/local/bin/
	mv srtm4/srtm4_which_tile /usr/local/bin/

clean:
	$(MAKE) -C srtm4 clean
