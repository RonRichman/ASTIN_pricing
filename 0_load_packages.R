# List of packages
packages <- c("data.table", "dplyr", "ggplot2", "ggpubr", 
              "keras", "tensorflow", "stringr", "lubridate")

# Function to install and load packages
install_and_load_packages <- function(packages) {
  for(package in packages) {
    # Check if the package is installed
    if (!require(package, character.only = TRUE)) {
      # Install the package if it's not already installed
      install.packages(package, repos = "http://cran.us.r-project.org")
      # Load the package
      library(package, character.only = TRUE)
    }
  }
}

# Use the function to install and load packages
install_and_load_packages(packages)
