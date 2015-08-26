local MNET_DIR = (path.getabsolute(path.join(path.getdirectory(_SCRIPT),"../..")) .. "/")
local SAMPLE_DIR = (path.getabsolute(path.getdirectory(_SCRIPT)) .. "/")

project "sample_helloworld"
	kind "ConsoleApp"
	location "%{sln.location}"
	language "C++"

	includedirs {
		MNET_DIR .. "include/",
	}

	files {
		SAMPLE_DIR .. "**",
	}

	links {
		"mnet",
	}

	configuration {}

