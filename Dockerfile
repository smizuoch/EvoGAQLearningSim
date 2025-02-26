FROM dorowu/ubuntu-desktop-lxde-vnc

# Install dependencies
# Add the Google Linux signing key
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -

# Now update and install packages
RUN apt-get update && apt-get install -y g++ make libsfml-dev

# Copy the source code
COPY . /app

RUN g++ /app/EvoGAQLearningSim.cpp -o sim -lsfml-graphics -lsfml-window -lsfml-system

# Set the working directory
WORKDIR /app

# port
EXPOSE 80

# Run the application
CMD ["./sim"]
