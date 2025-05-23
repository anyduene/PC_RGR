#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace std;

const string PYTHON_ENV = "/Users/illiamatsko/My/University/paralel/HyperparameterTuning/.venv/bin/python3";
const string PYTHON_SCRIPT = "/Users/illiamatsko/My/University/paralel/HyperparameterTuning/model/lstm.py";

vector<string> global_indicators = {"SMA", "EMA", "RSI", "MACD", "BB_UP", "BB_DOWN", "ATR", "High", "Low", "Open", "Volume"};

struct Parameter {
    double value;

    Parameter(double v) : value(v) {}
};

struct Chromosome {
    vector<string> indicators;
    Parameter layers;
    Parameter units;
    Parameter dropout;
    Parameter epochs;
    Parameter batch_size;
    Parameter window_size;
    double fitness;

    Chromosome() : layers(0), units(0), dropout(0), epochs(0),
                   batch_size(0), window_size(0), fitness(10000.0) {}

    void print() {
        cout << "Indicators: ";
        for (const auto& ind : indicators) {
            cout << ind << " ";
        }
        cout << "\nLayers: " << layers.value
             << "\nUnits: " << units.value
             << "\nDropout: " << dropout.value
             << "\nEpochs: " << epochs.value
             << "\nBatch Size: " << batch_size.value
             << "\nWindow Size: " << window_size.value
             << "\nFitness (Error): " << fitness << "%" << endl;
    }
};

class GeneticAlgorithm {
private:
    mt19937 gen{random_device{}()};
    int population_size;
    int generations;
    double mutation_rate;
    vector<Chromosome> population;

    const int MIN_LAYERS = 2; // 2
    const int MAX_LAYERS = 6; // 6
    const int MIN_UNITS = 40; // 40
    const int MAX_UNITS = 256; // 256
    const double MIN_DROPOUT = 0.05;
    const double MAX_DROPOUT = 0.5;
    const int MIN_EPOCHS = 30; // 30
    const int MAX_EPOCHS = 75; // 75
    const int MIN_BATCH_SIZE = 16;
    const int MAX_BATCH_SIZE = 128;
    const int MIN_WINDOW_SIZE = 10;
    const int MAX_WINDOW_SIZE = 90;

public:
    GeneticAlgorithm(int pop_size, int gen_count, double mut_rate) {
        population_size = pop_size;
        generations = gen_count;
        mutation_rate = mut_rate;

        initializePopulation();
    }

    void initializePopulation() {
        population.clear();
        population.reserve(population_size);

        for (int i = 0; i < population_size; ++i) {
            population.push_back(createRandomChromosome());
        }
    }

    Chromosome createRandomChromosome() {
        Chromosome chrom;

        uniform_int_distribution<> layers_dist(MIN_LAYERS, MAX_LAYERS);
        uniform_int_distribution<> units_dist(MIN_UNITS, MAX_UNITS);
        uniform_real_distribution<> dropout_dist(MIN_DROPOUT, MAX_DROPOUT);
        uniform_int_distribution<> epochs_dist(MIN_EPOCHS, MAX_EPOCHS);
        uniform_int_distribution<> batch_dist(MIN_BATCH_SIZE, MAX_BATCH_SIZE);
        uniform_int_distribution<> window_dist(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
        chrom.indicators = global_indicators;
        chrom.layers.value = layers_dist(gen);
        chrom.units.value = units_dist(gen);
        chrom.dropout.value = dropout_dist(gen);
        chrom.epochs.value = epochs_dist(gen);
        chrom.batch_size.value = batch_dist(gen);
        chrom.window_size.value = window_dist(gen);
        chrom.fitness = 100000.0;

        return chrom;
    }

    void createFiles() {
        for (int i = 0; i < population_size; i++) {
            Chromosome chrom = population[i];
            ofstream params_file("/Users/illiamatsko/My/University/paralel/GenAlg/params_" + to_string(i) + ".txt");
            string possible_indicators_str;
            for (const auto& ind : global_indicators) {
                possible_indicators_str += ind + " ";
            }
            string indicators_str;
            for (const auto& ind : chrom.indicators) {
                indicators_str += ind + " ";
            }

            params_file << "possible_indicators=" << possible_indicators_str << "\n"
                        << "indicators=" << indicators_str << "\n"
                        << "window_size=" << chrom.window_size.value << "\n"
                        << "epochs=" << chrom.epochs.value << "\n"
                        << "batch_size=" << chrom.batch_size.value << "\n"
                        << "layers=" << chrom.layers.value << "\n"
                        << "units=" << chrom.units.value << "\n"
                        << "dropout=" << chrom.dropout.value << "\n";
            params_file.close();

        }
    }

    string execPython(const string& command) {
        string result;
        char buffer[128];

        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) throw runtime_error("popen() failed!");

        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }

        int rc = pclose(pipe);
        if (rc != 0) {
            cerr << "Python script returned non-zero exit code: " << rc << endl;
            throw runtime_error("Python script execution failed!");
        }

        return result;
    }

    void evaluatePopulation() {
        createFiles();

        for (int i = 0; i < population_size; i++) {
            string command = PYTHON_ENV + " " + PYTHON_SCRIPT + " /Users/illiamatsko/My/University/paralel/GenAlg/params_" + to_string(i) + ".txt";
            string output = execPython(command);

            population[i].fitness = stod(output);
            cout << "Chromosome " << i << "/" << population_size-1 << ". Fitness: " << output << '%' << endl;
        }
    }

    Chromosome selectParent() {
        uniform_int_distribution<> dist(2, population_size-2);
        Chromosome candidate1 = population[dist(gen)];
        Chromosome candidate2 = population[dist(gen)];

        return (candidate1.fitness < candidate2.fitness) ? candidate1 : candidate2;
    }

    Chromosome crossover(const Chromosome& parent1, const Chromosome& parent2) {
        Chromosome child;
        uniform_real_distribution<> dist(0.0, 1.0);

        child.indicators = (dist(gen) < 0.5) ? parent1.indicators : parent2.indicators;

        child.layers = (dist(gen) < 0.5) ? parent1.layers : parent2.layers;
        child.units = (dist(gen) < 0.5) ? parent1.units : parent2.units;
        child.dropout = (dist(gen) < 0.5) ? parent1.dropout : parent2.dropout;
        child.epochs = (dist(gen) < 0.5) ? parent1.epochs : parent2.epochs;
        child.batch_size = (dist(gen) < 0.5) ? parent1.batch_size : parent2.batch_size;
        child.window_size = (dist(gen) < 0.5) ? parent1.window_size : parent2.window_size;

        child.fitness = 100000.0;

        return child;
    }


    Chromosome mutate(Chromosome chrom) {
        uniform_real_distribution<> dist(0.0, 1.0);

        if (dist(gen) < 1.0 / 6.0) {
            uniform_int_distribution<> layers_dist(MIN_LAYERS, MAX_LAYERS);
            chrom.layers.value = layers_dist(gen);
        }

        if (dist(gen) < 1.0 / 6.0) {
            uniform_int_distribution<> units_dist(MIN_UNITS, MAX_UNITS);
            chrom.units.value = units_dist(gen);
        }

        if (dist(gen) < 1.0 / 6.0) {
            uniform_real_distribution<> dropout_dist(MIN_DROPOUT, MAX_DROPOUT);
            chrom.dropout.value = dropout_dist(gen);
        }

        if (dist(gen) < 1.0 / 6.0) {
            uniform_int_distribution<> epochs_dist(MIN_EPOCHS, MAX_EPOCHS);
            chrom.epochs.value = epochs_dist(gen);
        }

        if (dist(gen) < 1.0 / 6.0) {
            uniform_int_distribution<> batch_dist(MIN_BATCH_SIZE, MAX_BATCH_SIZE);
            chrom.batch_size.value = batch_dist(gen);
        }

        if (dist(gen) < 1.0 / 6.0) {
            uniform_int_distribution<> window_dist(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
            chrom.window_size.value = window_dist(gen);
        }

        return chrom;
    }

    void run(double learning_rate) {
        cout << "Starting sequential genetic algorithm optimization..." << endl;
        cout << "Population size: " << population_size << ", Generations: " << generations << "\n\n";

        Chromosome best_overall;

        for (int generation = 0; generation < generations; generation++) {
            cout << "\n--- Generation " << generation + 1 << " ---" << endl;


            evaluatePopulation();

            sort(population.begin(), population.end(),
                 [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });

            cout << "\nBest chromosome in generation " << generation + 1 << ": ";
            population[0].print();

            if (generation == 0 || population[0].fitness < best_overall.fitness) {
                best_overall = population[0];
            }

            double avg_fitness = 0.0;
            double min_fitness = population[0].fitness;
            double max_fitness = population[population_size - 1].fitness;
            double range = max_fitness - min_fitness;
            if (range == 0) range = 1e-6;

            vector<double> normalized_fitness;
            for (auto& chrom : population) {
                avg_fitness += chrom.fitness;
                normalized_fitness.push_back((chrom.fitness - min_fitness) / range);
            }
            avg_fitness /= population_size;
            double normalized_avg_fitness = (avg_fitness - min_fitness) / range;



            vector<Chromosome> new_population;

            new_population.push_back(population[0]);
            new_population.push_back(population[1]);

            new_population.push_back(mutate(population[population_size - 1]));
            new_population.push_back(mutate(population[population_size - 2]));

            while (new_population.size() < population_size) {
                Chromosome parent1 = selectParent();
                Chromosome parent2 = selectParent();
                Chromosome child = crossover(parent1, parent2);
                new_population.push_back(child);
            }

            for(int i=4; i<population_size; i++) {
                uniform_real_distribution<> prob_dist(0.0, 1.0);

                if (prob_dist(gen) < mutation_rate) {
                    Chromosome mutated = mutate(population[i]);
                    new_population[i] = mutated;
                }
            }

            population = new_population;
        }

        sort(population.begin(), population.end(),
             [](const Chromosome& a, const Chromosome& b) { return a.fitness < b.fitness; });

        if (population[0].fitness < best_overall.fitness) {
            best_overall = population[0];
        }

        cout << "\n\n=== Best chromosome after all generations ===" << endl;
        best_overall.print();
    }
};

int main() {
    int population_size = 10;
    int generations = 15;
    double mutation_rate = 0.2;

    double learning_rate = 0.1;

    GeneticAlgorithm genalg = GeneticAlgorithm(population_size, generations, mutation_rate);
    genalg.run(learning_rate);
}