local MNET_DIR = (path.getabsolute(path.getdirectory(_SCRIPT)) .. "/")

project "mnet"
	kind "StaticLib"
	location "%{sln.location}"
	language "C++"

	includedirs {
		MNET_DIR .. "include/",

		MNET_DIR .. "src/",
	}

	files {
		MNET_DIR .. "include/**",
		MNET_DIR .. "src/core/**",
	}

	configuration "windows"
		files {
			MNET_DIR .. "src/plat_windows/**",
		}

	configuration {}
