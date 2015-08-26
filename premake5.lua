--
-- Premake 5.x build configuration script
-- Use this script to configure the project with Premake5.
--

solution "mnet_sln"
	configurations {"Release", "Debug"}
	location  (_OPTIONS["to"])
	warnings  "Extra"

	flags {
		"Symbols",
		"FatalWarnings",
		"NoMinimalRebuild",
		"NoIncrementalLink",
		"NoEditAndContinue",
	}

	configuration "Debug"
		targetdir "bin/debug"

	configuration "Release"
		targetdir "bin/release"
		optimize  "Speed"

	configuration {}

	dofile "mnet.premake5.lua"
	dofile "samples/helloworld/helloworld.lua"


--
-- Use the --to=path option to control where the project files get generated. I use
-- this to create project files for each supported toolset, each in their own folder,
-- in preparation for deployment.
--
	newoption {
		trigger     = "to",
		value       = "path",
		description = "Set the output location for the generated files"
	}
