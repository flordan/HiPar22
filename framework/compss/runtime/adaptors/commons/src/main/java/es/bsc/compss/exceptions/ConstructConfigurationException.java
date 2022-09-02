/*
 *  Copyright 2002-2022 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package es.bsc.compss.exceptions;

/**
 * Representation of an exception while constructing an Adaptor Configuration.
 */
public class ConstructConfigurationException extends Exception {

    /**
     * Exceptions Version UID are 2L in all Runtime.
     */
    private static final long serialVersionUID = 2L;


    /**
     * New empty Construct Configuration Exception.
     */
    public ConstructConfigurationException() {
        super();
    }

    /**
     * New Construct Configuration Exception with a nested exception @e.
     * 
     * @param e Causing exception
     */
    public ConstructConfigurationException(Exception e) {
        super(e);
    }

    /**
     * New Construct Configuration Exception with a message @msg.
     * 
     * @param msg Error message
     */
    public ConstructConfigurationException(String msg) {
        super(msg);
    }

}
