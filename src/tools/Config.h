/*
 * Config.h
 *
 *  Created on: Jul 2, 2019
 *      Author: robert
 */

#ifndef TOOLS_CONFIG_H_
#define TOOLS_CONFIG_H_


#include <map>

typedef std::map<std::string, std::string> config_section;
typedef std::map<std::string, config_section> config;

/**
 * @brief Global config. Values can be retrieved with the get function. The config file will be
 * loaded automically when the get function is called the first time. If you want it to be
 * efficient cache the values instead of reading getting them from the config
 * over and over.
 *
 * Supported datatypes and default values if the key can not be found in the config
 *
 * string = ""
 * long = 0
 * double = 0.0
 * bool = false
 *
 */
class Config {
public:
	/**
	 * Defaults to config.ini
	 */

	static Config* getConfig();

	template<class T>
	T get( const std::string& section, const std::string& key );

	void operator=( Config const& ) = delete;
	Config( Config const& ) = delete;
private:

	Config(){};

	std::string configFile = "config.ini" ;

	config c;

	void readConfig();

	static Config* mConfig;
};


#endif /* TOOLS_CONFIG_H_ */
